#![allow(clippy::upper_case_acronyms)]

use anyhow::{anyhow, Result};
use arrayfire::{constant, Array};
use tokenizers::Tokenizer;

use super::{
    model::{RWKVLayerState, RWKV},
    util::{Elem, ReqOps},
};

/// Context that holds the state of the RWKV model.
pub struct RWKVContext<T: ReqOps> {
    /// The RWKV model data — immutable.
    pub rwkv: RWKV<T>,
    /// Model state.
    pub state: Vec<RWKVLayerState<T>>,
    /// Probabilities from the last step (starts filled with zeros).
    pub last_probs: Array<T>,
    /// The tokenizer.
    pub tokenizer: Tokenizer,

    /// Number of vocabulary items.
    pub n_vocab: usize,
    /// Number of embedding items.
    pub n_embed: usize,
    /// Number of layers in the model.
    pub n_layers: usize,
}

impl RWKVContext<Elem> {
    pub fn new(rwkv: RWKV<Elem>, tokenizer: Tokenizer) -> Self {
        let dims = rwkv.emb.dims();
        let [n_vocab, n_embed, _, _] = dims.get();
        let n_layers = rwkv.layers.len();

        let initial_state = std::iter::repeat(RWKVLayerState::new(*n_embed as usize))
            .take(n_layers)
            .collect::<Vec<_>>();
        let initial_probs = constant!(0.0; *n_vocab);

        Self {
            n_embed: *n_embed as usize,
            n_layers,
            n_vocab: *n_vocab as usize,
            rwkv,
            state: initial_state,
            last_probs: initial_probs,
            tokenizer,
        }
    }

    /// Feeds some text to the model. A closure can be specified here to allow
    /// showing progress since it can take a while for large prompts/models.
    ///
    /// Evaluating the model generates probabilities, but they're not used here.
    pub fn feed_prompt<S: AsRef<str>>(&mut self, s: S, f: Option<impl Fn(String)>) -> Result<()> {
        let toks = self
            .tokenizer
            .encode(s.as_ref(), false)
            .map_err(|e| anyhow!(e))?;

        for tid in toks.get_ids().iter() {
            self.last_probs = self.rwkv.evaluate(*tid as usize, &mut self.state);
            if let Some(f) = &f {
                self.tokenizer
                    .decode(vec![*tid], false)
                    .map(f)
                    .map_err(|e| anyhow!(e))?;
            }
        }
        Ok(())
    }

    /// Infers the next token. Takes a closure that looks at the probabilities
    /// vector and figures out what token to pick.
    pub fn infer_next_token(
        &mut self,
        mut samplefun: impl FnMut(&Array<Elem>) -> Result<usize>,
    ) -> Result<Option<String>> {
        let tokid = samplefun(&self.last_probs)?;
        if tokid == 0 {
            return Ok(None);
        }
        let output = self
            .tokenizer
            .decode(vec![tokid as u32], false)
            .map_err(|e| anyhow!(e))?;
        self.last_probs = self.rwkv.evaluate(tokid, &mut self.state);
        Ok(Some(output))
    }
}
