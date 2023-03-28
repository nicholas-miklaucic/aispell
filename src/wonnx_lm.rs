//! `LM` implementation using the `edge-transformers` crate and ONNX.

use std::path::Path;
use std::str::from_utf8;
use std::sync::Mutex;
use std::{borrow::Cow, collections::HashMap};

use iced::futures::executor::block_on;
use lazy_static::lazy_static;
use tokenizers::tokenizer::Tokenizer;
use wonnx::utils::OutputTensor;
use wonnx::SessionConfig;
use wonnx::{utils::InputTensor, Session};

use tch::{IndexOp, Kind, Reduction, Tensor};

use crate::lm::LM;

pub struct WonnxLM {
    tokenizer: Tokenizer,
    model: Session,
    cache: Mutex<HashMap<String, Tensor>>,
}

impl WonnxLM {
    pub fn try_new_cached(model_id: &str) -> anyhow::Result<Self> {
        let models_path = Path::new("./models").join(model_id);
        let model_path = models_path.join("decoder_model_fixed.onnx");
        let tokenizer_path = models_path.join("tokenizer.json");
        // let special_path = models_path.join("special_tokens_map.json");

        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let model = block_on(Session::from_path_with_config(
            model_path,
            &SessionConfig::new().with_outputs(Some(vec!["logits".to_string()])),
        ))?;

        Ok(Self {
            tokenizer,
            model,
            cache: Default::default(),
        })
    }

    pub(crate) fn compute_logits_uncached(&self, text: &str) -> Tensor {
        dbg!(text);
        let tokens: Vec<i64> = self
            .tokenizer
            .encode(text, true)
            .unwrap()
            .get_ids()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let input_ids: InputTensor = InputTensor::I64(Cow::from(tokens.as_slice()));
        let mut in_map = HashMap::new();
        in_map.insert("input_ids".into(), input_ids);
        let outs = block_on(self.model.run(&in_map)).unwrap();
        dbg!(outs.keys());

        let model_output_vec: Vec<f64> = match outs.get("logits").unwrap() {
            OutputTensor::F32(vec) => vec.iter().map(|&x| x as f64).collect(),
            OutputTensor::I32(vec) => vec.iter().map(|&x| x as f64).collect(),
            OutputTensor::I64(vec) => vec.iter().map(|&x| x as f64).collect(),
            OutputTensor::U8(vec) => vec.iter().map(|&x| x as f64).collect(),
        };

        // dbg!(model_output.lm_logits.size());
        let model_output = Tensor::from(model_output_vec.as_slice());
        model_output.swapaxes(1, 2)
    }
}

impl LM<u8> for WonnxLM {
    fn loss(&self, context: &[u8], completion: &[u8]) -> f64 {
        let text = from_utf8(context).unwrap().to_owned() + from_utf8(completion).unwrap();
        let tokens = self.tokenizer.encode(text, true).unwrap();

        let input_str = self
            .tokenizer
            .decode(tokens.get_ids()[..tokens.len() - 1].to_vec(), true)
            .unwrap();

        let input_ids: Tensor = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i64)
            .collect::<Vec<i64>>()
            .as_slice()
            .into();

        let map = &mut *self.cache.lock().unwrap();

        let logits = map
            .entry(input_str)
            .or_insert_with_key(|k| self.compute_logits_uncached(k));

        // dbg!(logits.size(), input_ids.size());

        let loss: f64 = logits
            .cross_entropy_loss::<Tensor>(
                &input_ids.i(1..).view((1, -1)),
                None,
                Reduction::None,
                -100,
                0.0,
            )
            .sum_dim_intlist([1_i64].as_slice(), false, Kind::Float)
            .into();

        loss
    }
}

lazy_static! {
    pub static ref BLOOM: Mutex<WonnxLM> = WonnxLM::try_new_cached("bloom-560m").unwrap().into();
    // pub static ref LLM: Mutex<GptNeoLM> = GptNeoLM::try_new().unwrap().into();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prob_bloom() {
        let ctxt = b"The cat is out of the ";
        let lm = BLOOM.lock().unwrap();
        let loss1 = lm.loss(ctxt, b"bag");
        let loss2 = lm.loss(ctxt, b"bug");
        assert!(loss1 < loss2, "\n{:.3} < {:.3} failed", loss1, loss2);
    }
}
