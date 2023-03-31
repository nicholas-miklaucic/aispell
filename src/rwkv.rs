//! RWKV implementation.

/// Context related functions. Holds the model state and and last probabilities vector.
pub mod context;
/// Functions related to loading the model from disk.
pub mod loader;
/// The actual model and code related to evaluating it.
pub mod model;
/// Utility functions.
pub mod util;

use std::{collections::HashMap, path::Path, str::from_utf8, sync::Mutex};

use anyhow::Result;
use lazy_static::lazy_static;
use ndarray::{s, Array2, Axis};
use tch::{IndexOp, Kind, Reduction, Tensor};
use tokenizers::Tokenizer;

use crate::{
    lm::LM,
    rwkv::{model::RWKVLayerState, util::pardot},
};

use self::{model::RWKV, util::mmap_file};

/// LM using BlinkDL's RWKV.
#[derive(Debug)]
pub struct RwkvLM {
    /// The tokenizer.
    tokenizer: Tokenizer,
    /// The model itself.
    model: RWKV<f32>,
    /// The cache of computed states.
    cache: Mutex<HashMap<String, Tensor>>,
}

impl RwkvLM {
    pub fn try_new() -> Result<Self> {
        let models_path = Path::new("./models/rwkv-430m");
        let model_path = models_path.join("RWKV-4-Pile-430M-20220808-8066.safetensors");
        let tokenizer_path = models_path.join("20B_tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let rwkv = mmap_file(model_path.display().to_string().as_str())?.try_into()?;

        Ok(Self {
            tokenizer,
            model: rwkv,
            cache: Default::default(),
        })
    }

    pub(crate) fn compute_logits_uncached(&self, text: &str) -> Tensor {
        dbg!(text);
        let rwkv = &self.model;
        let tokens: Vec<u32> = self
            .tokenizer
            .encode(text, true)
            .unwrap()
            .get_ids()
            .to_vec();

        let n_tokens = tokens.len();

        let [n_vocab, n_embed] = rwkv.emb.shape() else { panic!("Shape is wrong")};
        let n_layers = rwkv.layers.len();
        let mut state = std::iter::repeat(RWKVLayerState::new(*n_embed))
            .take(n_layers)
            .collect::<Vec<_>>();

        let mut logits = Array2::default([n_tokens, *n_vocab]);
        for (i, token) in tokens.into_iter().enumerate() {
            let initial_x = rwkv.ln0.norm(&rwkv.emb.index_axis(Axis(0), token as usize));

            let x = rwkv
                .layers
                .iter()
                .enumerate()
                .fold(initial_x, |x, (lnum, layer)| {
                    rwkv.evaluate_layer(x, layer, &mut state[lnum])
                });

            let x = pardot(&rwkv.head, &rwkv.ln_out.norm(&x.view()));

            logits.slice_mut(s![i, ..]).assign(&x);
        }

        let mut model_output = logits.insert_axis(Axis(0));
        model_output.swap_axes(1, 2);
        model_output.as_standard_layout().try_into().unwrap()
    }
}

impl LM<u8> for RwkvLM {
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
    pub static ref RWKV_430M: Mutex<RwkvLM> = RwkvLM::try_new().unwrap().into();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prob_rwkv() {
        let ctxt = b"The cat is out of the ";
        let lm = RWKV_430M.lock().unwrap();
        let loss1 = lm.loss(ctxt, b"bag");
        let loss2 = lm.loss(ctxt, b"bug");
        assert!(loss1 < loss2, "\n{:.3} < {:.3} failed", loss1, loss2);
    }
}
