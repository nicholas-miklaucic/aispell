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
use arrayfire::{exp, index, max, sigmoid, sum, Array};
use lazy_static::lazy_static;
use tokenizers::Tokenizer;

use crate::{
    lm::LM,
    rwkv::{model::RWKVLayerState, util::pardot},
};

use self::{
    model::RWKV,
    util::{mmap_file, ReqOps},
};

/// The RWKV model state, including a loss.
#[derive(Clone)]
pub struct LossState<T: ReqOps> {
    /// The previous tokens.
    pub tokens: Vec<u32>,
    /// The output logits for the next token. None if no prior state exists.
    pub logits: Option<Array<T>>,
    /// The model state.
    pub state: Vec<RWKVLayerState<T>>,
    /// The loss.
    pub loss: f64,
}

pub fn softmax<T: ReqOps>(x: Array<T>) -> Array<T> {
    let exp_x = exp(&(x - max(&x, 0)));
    exp_x / sum(&exp_x)
}

/// Like PyTorch item(), converts [x] to x. Panics if array is bigger than 1
/// element.
pub fn item<T: ReqOps>(x: Array<T>) -> T {
    assert_eq!(
        x.elements(),
        1,
        "item(): Input {} is not one-dimensional",
        x.dims()
    );

    let mut out = vec![Default::default(); x.elements()];
    x.host(&mut out);
    out[0]
}

impl<T: ReqOps> LossState<T> {
    /// Initializes an empty state.
    pub fn new_empty(n_layers: usize, n_embed: usize) -> Self {
        Self {
            tokens: Default::default(),
            logits: None,
            state: std::iter::repeat(RWKVLayerState::new(n_embed))
                .take(n_layers)
                .collect(),
            loss: 0.0,
        }
    }

    /// Processes a new list of tokens.
    pub fn process_state(&mut self, tokens: &[u32], rwkv: &RWKV<T>) -> Self {
        let [n_vocab, n_embed, _, _] = rwkv.emb.dims().get();
        let n_layers = rwkv.layers.len();

        for token in tokens.iter() {
            self.loss += match self.logits {
                Some(x) => -f64::from(index(&softmax(x), token)).log(),
                None => 0.0,
            };
            let initial_x = rwkv.ln0.norm(&rwkv.emb[token]);

            let x = rwkv
                .layers
                .iter()
                .enumerate()
                .fold(initial_x, |x, (lnum, layer)| {
                    rwkv.evaluate_layer(x, layer, &mut self.state[lnum])
                });

            self.logits = pardot(&rwkv.head, &rwkv.ln_out.norm(&x.view()));
            self.tokens.push(token);
        }

        self
    }
}

/// LM using BlinkDL's RWKV.
#[derive(Debug)]
pub struct RwkvLM<T: ReqOps> {
    /// The tokenizer.
    tokenizer: Tokenizer,
    /// The model itself.
    model: RWKV<f32>,
    /// The cache of computed states.
    cache: Mutex<HashMap<Vec<u32>, LossState<T>>>,
}

impl<T: ReqOps> RwkvLM<T> {
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
}

impl<T: ReqOps> LM<u8> for RwkvLM<T> {
    fn loss(&self, context: &[u8], completion: &[u8]) -> f64 {
        let rwkv = &self.model;
        let [n_vocab, n_embed, _, _] = rwkv.emb.dims().get();
        let n_layers = rwkv.layers.len();

        let pretext = from_utf8(context).unwrap().to_owned();
        let text = from_utf8(completion).unwrap().to_owned();
        let pre_tokens = self.tokenizer.encode(pretext, true).unwrap().get_ids();
        let tokens = self.tokenizer.encode(text, true).unwrap().get_ids();

        let map = *self.cache.lock().unwrap();

        let pre_state = map.entry(context.to_vec()).or_insert_with_key(|k| {
            let mut st = LossState::new_empty(n_layers, n_embed);
            st.process_state(pre_tokens, rwkv)
        });

        let state = pre_state;
        let mut curr_tokens = pre_tokens.clone();
        for token in tokens {
            curr_tokens.push(token);
            let state = map.entry(curr_tokens).or_insert_with_key(|k| {
                state.clone().process_state(token, rwkv);
            });
        }

        state.loss
    }
}

lazy_static! {
    pub static ref RWKV_430M: Mutex<RwkvLM<f32>> = RwkvLM::try_new().unwrap().into();
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
