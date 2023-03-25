//! Defines the core `LM` trait and an implementation thereof.

use cached::proc_macro::cached;
use cached::SizedCache;
use std::collections::HashMap;
use std::str::from_utf8;
use std::sync::Mutex;

use lazy_static::lazy_static;
use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
use rust_bert::gpt2::{
    Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources,
};
use rust_bert::pipelines::generation_utils::{Cache, LMHeadModel, LMModelOutput};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tch::kind::Kind::Float;
use tch::{nn, no_grad, Device, IndexOp, Reduction, Tensor};

/// A language model that gives probabilities for words in context, with letters of type `T`.
pub trait LM<T> {
    /// Gets the loss of the given completion following the given context.
    fn loss(&self, context: &[T], completion: &[T]) -> f64;
}

/// A GPT2 language model.
pub struct GptLM {
    tokenizer: Gpt2Tokenizer,
    model: GPT2LMHeadModel,
    cache: Mutex<HashMap<String, Tensor>>,
}

impl GptLM {
    pub fn try_new() -> anyhow::Result<Self> {
        let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
        let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
        let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
        let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

        let config_path = config_resource.get_local_path()?;
        let vocab_path = vocab_resource.get_local_path()?;
        let merges_path = merges_resource.get_local_path()?;
        let weights_path = model_resource.get_local_path()?;

        //    Set-up model
        let device = Device::Cpu;
        let mut vs = nn::VarStore::new(device);
        let tokenizer: Gpt2Tokenizer = Gpt2Tokenizer::from_file(
            vocab_path.to_str().unwrap(),
            merges_path.to_str().unwrap(),
            false,
        )?;
        let config = Gpt2Config::from_file(config_path);
        let model = GPT2LMHeadModel::new(vs.root(), &config);
        vs.load(weights_path)?;

        Ok(Self {
            tokenizer,
            model,
            cache: HashMap::default().into(),
        })
    }

    fn compute_logits_uncached(&self, text: &str) -> Tensor {
        let tokens = self
            .tokenizer
            .encode(&text, None, 1000, &TruncationStrategy::DoNotTruncate, 0)
            .token_ids;

        let input_ids = Tensor::of_slice(&tokens).view((1, -1));
        let model_output: LMModelOutput = no_grad(|| {
            self.model.forward_t(
                Some(&input_ids),
                Cache::None,
                None,
                None,
                None,
                None,
                None,
                None,
                false,
            )
        })
        .unwrap();

        // dbg!(model_output.lm_logits.size());
        model_output.lm_logits.swapaxes(1, 2)
    }
}

impl LM<u8> for GptLM {
    fn loss(&self, context: &[u8], completion: &[u8]) -> f64 {
        let text = from_utf8(context).unwrap().to_owned() + from_utf8(completion).unwrap();
        let tokens = self
            .tokenizer
            .encode(&text, None, 1000, &TruncationStrategy::DoNotTruncate, 0)
            .token_ids;

        let input_str = self
            .tokenizer
            .convert_tokens_to_string(self.tokenizer.tokenize(&text)[..tokens.len() - 1].to_vec());

        let input_ids = Tensor::of_slice(&tokens).view((1, -1));

        let map = &mut *self.cache.lock().unwrap();
        let logits = map
            .entry(input_str)
            .or_insert_with_key(|k| self.compute_logits_uncached(k));
        // dbg!(logits.size(), input_ids.size(), &text);

        let loss: f64 = logits
            .cross_entropy_loss::<Tensor>(&input_ids.i((.., 1..)), None, Reduction::None, -100, 0.0)
            .sum_dim_intlist([1_i64].as_slice(), false, Float)
            .into();

        loss
    }
}

lazy_static! {
    pub static ref LLM: Mutex<GptLM> = GptLM::try_new().unwrap().into();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prob() {
        let ctxt = b"The cat is out of the ";
        let lm = LLM.lock().unwrap();
        let loss1 = lm.loss(ctxt, b"bag");
        let loss2 = lm.loss(ctxt, b"bug");
        assert!(loss1 < loss2, "\n{:.3} < {:.3} failed", loss1, loss2);
    }
}
