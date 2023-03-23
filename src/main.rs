use anyhow;

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

fn main() -> anyhow::Result<()> {
    //    Set-up model
    //    Resources paths
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

    let input = vec![
        "Can anyone link me to a snipping otol",
        "Can anyone link me to a snipping tool",
    ];

    let tokenized_input =
        tokenizer.encode_list(&input, 1000, &TruncationStrategy::DoNotTruncate, 0);

    let max_len = tokenized_input
        .iter()
        .map(|input| input.token_ids.len())
        .max()
        .unwrap();

    const SPECIAL_TOKEN: i64 = 50256;
    let tokenized_input = tokenized_input
        .iter()
        .map(|input| input.token_ids.clone())
        .map(|mut input| {
            input.extend(vec![SPECIAL_TOKEN; max_len - input.len()]);
            input
        })
        .map(|input| Tensor::of_slice(&(input)))
        .collect::<Vec<_>>();
    let input_tensor = Tensor::stack(tokenized_input.as_slice(), 0).to(device);

    let mut past: Vec<Tensor> = Vec::with_capacity(config.n_layer as usize);
    for _ in 0..config.n_layer as usize {
        past.push(Tensor::rand(
            &[2, 1, config.n_head, 1, config.n_embd / config.n_head],
            (Float, device),
        ))
    }
    //    Forward pass
    let model_output: LMModelOutput = no_grad(|| {
        model.forward_t(
            Some(&input_tensor),
            Cache::None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
        )
    })?;

    let probs = model_output.lm_logits.i((0, -3, ..)).softmax(-1, Float);
    dbg!(model_output.lm_logits.size());
    dbg!(input_tensor.size());
    let (xx, ii) = probs.sort(0, true);
    let top_n = 10;

    for i in 0..top_n {
        println!(
            "{:6.2}% {}",
            f64::from(xx.i(i)) * 100.0,
            tokenizer.decode(&[ii.i(i).into()], false, true)
        );
    }

    println!(
        "{:?}",
        model_output
            .lm_logits
            .swapaxes(1, 2)
            .cross_entropy_loss::<Tensor>(&input_tensor, None, Reduction::None, SPECIAL_TOKEN, 0.0)
            .mean_dim([1_i64].as_slice(), false, Float)
    );
    Ok(())
}
