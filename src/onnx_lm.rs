//! `LM` implementation using the `edge-transformers` crate and ONNX.

use std::collections::HashMap;
use std::path::Path;
use std::str::from_utf8;
use std::sync::Mutex;

use edge_transformers::hf_hub::hf_hub_download;
use edge_transformers::ort::GraphOptimizationLevel;
use edge_transformers::tokenizer::AutoTokenizer;
use edge_transformers::{ConditionalGenerationModel, Device};
use lazy_static::lazy_static;

use ndarray::{Array2, Array3};
use ort::environment::Environment;
use ort::LoggingLevel;
use tch::{IndexOp, Kind, Reduction, Tensor};

use crate::lm::LM;

pub struct OnnxLM {
    tokenizer: AutoTokenizer,
    model: ConditionalGenerationModel<'static>,
    cache: Mutex<HashMap<String, Tensor>>,
}

impl OnnxLM {
    pub fn try_new_cached(model_id: &str) -> anyhow::Result<Self> {
        let environment = Environment::builder()
            .with_name("test")
            .with_log_level(LoggingLevel::Verbose)
            .build()?;

        let models_path = Path::new("./models").join(model_id);
        let model_path = models_path.join("decoder_model.onnx");
        let tokenizer_path = models_path.join("tokenizer.json");
        let special_path = models_path.join("special_tokens_map.json");

        let tokenizer = AutoTokenizer::new(tokenizer_path, special_path)?;
        let model = ConditionalGenerationModel::new_from_file(
            environment.into(),
            model_path,
            Device::CUDA,
            GraphOptimizationLevel::Level3,
        )?;

        Ok(Self {
            tokenizer,
            model,
            cache: Default::default(),
        })
    }
    pub fn try_new_hf(model_id: &str) -> anyhow::Result<Self> {
        let environment = Environment::builder()
            .with_name("test")
            .with_log_level(LoggingLevel::Verbose)
            .build()?;

        let model_path = hf_hub_download(&model_id, "decoder_model.onnx", None, None)?;
        let tokenizer_path = hf_hub_download(&model_id, "tokenizer.json", None, None)?;
        let special_tokens_path = hf_hub_download(&model_id, "special_tokens_map.json", None, None);

        let special_path = match special_tokens_path {
            Ok(p) => Ok(p),
            Err(_e) => hf_hub_download(&model_id, "config.json", None, None),
        }?;

        let tokenizer = AutoTokenizer::new(tokenizer_path, special_path)?;
        let model = ConditionalGenerationModel::new_from_file(
            environment.into(),
            model_path,
            Device::CPU,
            GraphOptimizationLevel::Level3,
        )?;

        Ok(Self {
            tokenizer,
            model,
            cache: Default::default(),
        })
    }

    pub(crate) fn compute_logits_uncached(&self, text: &str) -> Tensor {
        dbg!(text);
        let tokens = self
            .tokenizer
            .tokenizer
            .encode(text, true)
            .unwrap()
            .get_ids()
            .to_vec();

        let input_ids: Array2<u32> =
            Array2::from_shape_vec((1, tokens.len()), tokens.to_vec()).unwrap();
        let mut model_output: Array3<f32> = self.model.forward(input_ids, None, None).unwrap();

        // dbg!(model_output.lm_logits.size());
        model_output.swap_axes(1, 2);
        model_output.as_standard_layout().try_into().unwrap()
    }
}

impl LM<u8> for OnnxLM {
    fn loss(&self, context: &[u8], completion: &[u8]) -> f64 {
        let text = from_utf8(context).unwrap().to_owned() + from_utf8(completion).unwrap();
        let tokens = self.tokenizer.tokenizer.encode(text, true).unwrap();

        let input_str = self
            .tokenizer
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
    pub static ref BLOOM: Mutex<OnnxLM> = OnnxLM::try_new_cached("bloom-560m").unwrap().into();
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
