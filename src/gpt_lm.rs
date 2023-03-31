use std::collections::HashMap;
use std::marker::PhantomData;
use std::str::from_utf8;
use std::sync::Mutex;

use lazy_static::lazy_static;
use rust_bert::gpt2::{GPT2LMHeadModel, Gpt2Config};
use rust_bert::gpt_neo::{GptNeoConfig, GptNeoForCausalLM};
use rust_bert::pipelines::generation_utils::{Cache, LMHeadModel, LMModelOutput};
use rust_bert::resources::{RemoteResource, ResourceProvider};
use rust_bert::Config;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{Gpt2Vocab, Vocab};
use tch::kind::Kind::Float;
use tch::{nn, no_grad, Device, IndexOp, Reduction, Tensor};

macro_rules! gpt2_resources {
    ($name:ident) => {{
        use rust_bert::gpt2::{
            Gpt2ConfigResources, Gpt2MergesResources, Gpt2ModelResources, Gpt2VocabResources,
        };
        (
            Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::$name)),
            Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::$name)),
            Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::$name)),
            Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::$name)),
        )
    }};
}

macro_rules! gpt_neo_resources {
    ($name:ident) => {{
        use rust_bert::gpt_neo::{
            GptNeoConfigResources, GptNeoMergesResources, GptNeoModelResources,
            GptNeoVocabResources,
        };
        (
            Box::new(RemoteResource::from_pretrained(
                GptNeoConfigResources::$name,
            )),
            Box::new(RemoteResource::from_pretrained(GptNeoVocabResources::$name)),
            Box::new(RemoteResource::from_pretrained(
                GptNeoMergesResources::$name,
            )),
            Box::new(RemoteResource::from_pretrained(GptNeoModelResources::$name)),
        )
    }};
}

pub type GptLM = CausalLM<Gpt2Vocab, Gpt2Tokenizer, GPT2LMHeadModel>;
pub type GptNeoLM = CausalLM<Gpt2Vocab, Gpt2Tokenizer, GptNeoForCausalLM>;

/// A causal language model.
pub struct CausalLM<V: Vocab, T: Tokenizer<V>, M: LMHeadModel> {
    tokenizer: T,
    model: M,
    cache: Mutex<HashMap<String, Tensor>>,
    _vocab: PhantomData<V>,
}

impl GptLM {
    pub fn try_new() -> anyhow::Result<Self> {
        let (config_resource, vocab_resource, merges_resource, model_resource) =
            gpt2_resources!(GPT2);
        // let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
        // let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
        // let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
        // let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

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
            _vocab: Default::default(),
        })
    }
}

impl GptNeoLM {
    pub fn try_new() -> anyhow::Result<Self> {
        let (config_resource, vocab_resource, merges_resource, model_resource) =
            gpt_neo_resources!(GPT_NEO_125M);
        // let config_resource = Box::new(RemoteResource::from_pretrained(Gpt2ConfigResources::GPT2));
        // let vocab_resource = Box::new(RemoteResource::from_pretrained(Gpt2VocabResources::GPT2));
        // let merges_resource = Box::new(RemoteResource::from_pretrained(Gpt2MergesResources::GPT2));
        // let model_resource = Box::new(RemoteResource::from_pretrained(Gpt2ModelResources::GPT2));

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
        let config = GptNeoConfig::from_file(config_path);
        let model = GptNeoForCausalLM::new(vs.root(), &config).unwrap();
        vs.load(weights_path)?;

        Ok(Self {
            tokenizer,
            model,
            cache: HashMap::default().into(),
            _vocab: Default::default(),
        })
    }
}

impl<V: Vocab, T: Tokenizer<V>, M: LMHeadModel> CausalLM<V, T, M> {
    pub(crate) fn compute_logits_uncached(&self, text: &str) -> Tensor {
        dbg!(text);
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

impl<V: Vocab, T: Tokenizer<V>, M: LMHeadModel> LM<u8> for CausalLM<V, T, M> {
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
    // pub static ref LLM: Mutex<GptNeoLM> = GptNeoLM::try_new().unwrap().into();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logits() {
        let lm = LLM.lock().unwrap();
        let logits = lm.compute_logits_uncached(".").view(-1);
        let (_xx, ii) = logits.sort(-1, true);

        let answers = vec![198, 383, 314, 366, 357];
        let res: Vec<i64> = ii.i(..answers.len() as i64).into();
        for i in 0..5 {
            assert_eq!(res[i], answers[i], "\n{:#?}\n{:#?}", res, answers);
        }
    }

    #[test]
    fn test_prob() {
        let ctxt = b"The cat is out of the ";
        let lm = LLM.lock().unwrap();
        let loss1 = lm.loss(ctxt, b"bag");
        let loss2 = lm.loss(ctxt, b"bug");
        assert!(loss1 < loss2, "\n{:.3} < {:.3} failed", loss1, loss2);
    }
}
