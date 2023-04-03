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
use tch::{IndexOp, Kind, Tensor};
use tokenizers::Tokenizer;

use crate::{
    lm::LM,
    rwkv::{model::RWKVLayerState, util::pardot},
};

use self::{model::RWKV, util::mmap_file};

/// The RWKV model state, including a loss.
pub struct LossState {
    /// The previous tokens.
    pub tokens: Vec<u32>,
    /// The output logits for the next token. None if no prior state exists.
    pub logits: Option<Tensor>,
    /// The model state.
    pub state: Vec<RWKVLayerState>,
    /// The loss.
    pub loss: f64,
}

impl Clone for LossState {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            logits: self.logits.as_ref().map(|x| x.copy()),
            state: self.state.clone(),
            loss: self.loss.clone(),
        }
    }
}

pub fn softmax(x: &Tensor) -> Tensor {
    x.softmax(-1, Kind::Float)
}

/// Like PyTorch item(), converts [x] to x. Panics if array is bigger than 1
/// element.
pub fn item(x: Tensor) -> f64 {
    assert_eq!(
        x.numel(),
        1,
        "item(): Input {:?} is not one-dimensional",
        x.size()
    );

    let arr = x
        .flatten(0, x.dim() as i64)
        .iter::<f64>()
        .unwrap()
        .take(1)
        .next();
    arr.expect("item() of empty tensor")
}

impl LossState {
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
    pub fn process_state(mut self, tokens: &[u32], rwkv: &RWKV) -> Self {
        for token in tokens.iter() {
            self.loss += match &self.logits {
                Some(logits) => (dbg!(logits.i(*token as i64))
                    - dbg!(logits.logsumexp(&[-1], true)))
                .try_into()
                .unwrap(),
                None => 0.0,
            } as f64;
            dbg!(self.loss, rwkv.emb.size());
            let initial_x = rwkv.ln0.norm(&rwkv.emb.i(*token as i64));

            let x = rwkv
                .layers
                .iter()
                .enumerate()
                .fold(initial_x, |x, (lnum, layer)| {
                    // dbg!(x.size(), self.state[lnum].tm_state.size());
                    rwkv.evaluate_layer(x, layer, &mut self.state[lnum])
                });

            self.logits = Some(pardot(&rwkv.head, &rwkv.ln_out.norm(&x)));
            self.tokens.push(*token);
        }

        self
    }
}

/// LM using BlinkDL's RWKV.
pub struct RwkvLM {
    /// The tokenizer.
    tokenizer: Tokenizer,
    /// The model itself.
    model: RWKV,
    /// The cache of computed states.
    cache: Mutex<HashMap<Vec<u32>, LossState>>,
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
}

impl LM<u8> for RwkvLM {
    fn loss(&self, context: &[u8], completion: &[u8]) -> f64 {
        let rwkv = &self.model;
        let n_layers = rwkv.layers.len();

        let pretext = from_utf8(context).unwrap().to_owned();
        let text = from_utf8(completion).unwrap().to_owned();
        let pre_tokens = self.tokenizer.encode(pretext, true).unwrap();
        let tokens = self.tokenizer.encode(text, true).unwrap();

        let mut map = self.cache.lock().unwrap();

        map.entry(pre_tokens.get_ids().to_vec())
            .or_insert_with_key(|k| {
                let st = LossState::new_empty(n_layers, rwkv.emb.size()[1] as usize);
                st.process_state(k, rwkv)
            });

        let mut curr_tokens = pre_tokens.get_ids().clone().to_vec();
        for token in tokens.get_ids() {
            let prev_st = map.get(&curr_tokens).unwrap().clone();
            curr_tokens.push(*token);
            if let None = map.get(&curr_tokens) {
                map.insert(
                    curr_tokens.clone(),
                    prev_st.process_state(vec![*token].as_slice(), rwkv),
                );
            }
        }

        for k in map.keys() {
            debug_assert_eq!(k, &map.get(k).unwrap().tokens);
        }

        dbg!(&curr_tokens);
        map.get(&curr_tokens).unwrap().loss
    }
}

lazy_static! {
    pub static ref RWKV_430M: Mutex<RwkvLM> = RwkvLM::try_new().unwrap().into();
}

#[cfg(test)]
mod tests {
    use tch::Device;

    use super::*;

    #[test]
    fn test_rwkv_state() {
        let lm = RWKV_430M.lock().unwrap();
        dbg!(lm.model.emb.size());
    }

    #[test]
    fn test_torch() {
        let x = Tensor::arange(5, (Kind::Float, Device::Cpu));
        let zero = Tensor::zeros(&[1], (Kind::Float, Device::Cpu));
        assert_eq!(
            (&x - 2.0).max_other(&zero).square(),
            (&x - 2.0).relu().square()
        );

        // assert_eq!(
        //     x.std_mean_dim(vec![-1].as_slice(), false, true).0,
        //     x.std_mean(false).0
        // )
        dbg!(&x.i((1..,)).reshape(&[2, 2]).i(1), &x.i((3..,)));

        assert_eq!(
            pardot(&x.i((1..,)).reshape(&[2, 2]), &x.i((3..,))),
            Tensor::of_slice(vec![(1.0 * 3.0 + 2.0 * 4.0), (3.0 * 3.0 + 4.0 * 4.0)].as_slice())
        );
    }

    #[test]
    fn test_loss_rwkv() {
        let lm = RWKV_430M.lock().unwrap();
        let st = LossState::new_empty(lm.model.layers.len(), lm.model.emb.size()[1] as usize);
        let st = st.process_state(vec![510, 5349, 273, 6181, 310, 7785].as_slice(), &lm.model);
        let out_mean = item(st.logits.unwrap().mean(Kind::Float));
        assert!(
            (out_mean - (-13.1)).abs() < 0.1,
            "Out mean not 13.1: {}",
            out_mean
        );
        assert!(
            (st.loss - 21.6).abs() < 0.1,
            "Incorrect loss {}\n{:#?}",
            st.loss,
            st.tokens
        );
    }

    #[test]
    fn test_prob_rwkv() {
        let ctxt = b"The cat is out of the ";
        let lm = RWKV_430M.lock().unwrap();
        let loss1 = lm.loss(ctxt, b"bag");
        let loss2 = lm.loss(ctxt, b"bug");
        assert!(loss1 < loss2, "\n{} < {} failed", loss1, loss2);
    }
}
