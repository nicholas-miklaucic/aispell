//! Defines the core `LM` trait and an implementation thereof.

use std::str::from_utf8;

use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
use rust_bert::gpt2::{
    Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources,
};
use rust_bert::pipelines::generation_utils::{Cache, LMHeadModel, LMModelOutput};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use tch::kind::Kind::Float;
use tch::{nn, no_grad, Device, Reduction, Tensor};

/// A language model that gives probabilities for words in context, with letters of type `T`.
pub trait LM<T> {
    /// Gets the loss of the given completion following the given context.
    fn loss(&self, context: &[T], completion: &[T]) -> f64;
}

/// A GPT2 language model.
pub struct GptLM {
    tokenizer: Gpt2Tokenizer,
    model: GPT2LMHeadModel,
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

        Ok(Self { tokenizer, model })
    }
}

impl LM<u8> for GptLM {
    fn loss(&self, context: &[u8], completion: &[u8]) -> f64 {
        let text = from_utf8(context).unwrap().to_owned() + from_utf8(completion).unwrap();
        let tokens =
            self.tokenizer
                .encode(&text, None, 1000, &TruncationStrategy::DoNotTruncate, 0);

        let input_ids = Tensor::of_slice(&tokens.token_ids).unsqueeze(0);
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

        dbg!(model_output.lm_logits.size());

        let loss: f64 = model_output
            .lm_logits
            .swapaxes(1, 2)
            .cross_entropy_loss::<Tensor>(&input_ids, None, Reduction::None, -100, 0.0)
            .mean_dim([1_i64].as_slice(), false, Float)
            .into();

        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prob() {
        let ctxt = b"Jackson is the capital of ";
        let lm = GptLM::try_new().unwrap();
        let loss1 = lm.loss(ctxt, b"Mississippi");
        let loss2 = lm.loss(ctxt, b"Mississipi");
        assert!(loss1 < loss2, "\n{:.3} < {:.3} failed", loss1, loss2);
    }
}
