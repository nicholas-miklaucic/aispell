use anyhow::{anyhow, Error, Result};
use mmap_rs::Mmap;
use safetensors::{tensor::TensorView, SafeTensors};

use std::collections::HashMap;

use super::{
    model::*,
    util::{tensor_to_array1, tensor_to_array2},
};

/// LayerMap helper type to avoid repetition.
type LM<'a> = HashMap<String, safetensors::tensor::TensorView<'a>>;

/// Helper function for extracting a tensor from the HashMap by string key.
/// Takes a closure to convert from the SafeTensors TensorView struct to
/// a usable format.
fn gk<O>(m: &LM, k: &str, f: impl Fn(&TensorView) -> O) -> Result<O> {
    m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
}

/// Convert from a mmap (just a chunk of bytes) to the RWKV<T> struct
/// Requires the ConvertBF16Tensor trait (from `crate::utils`) due to
/// tensors being stored in bfloat16 format which isn't suitable for
/// actual calculation.
impl TryFrom<Mmap> for RWKV {
    type Error = Error;

    fn try_from(value: Mmap) -> std::result::Result<Self, Self::Error> {
        // Note that this actually just reads the metadata and not
        // the tensor data itself.
        let st = SafeTensors::deserialize(value.as_slice())?;
        // Use the TryFrom instance to convert from SafeTensors to RWKV<T>.
        (&st).try_into()
    }
}

impl TryFrom<(usize, &LM<'_>)> for LayerNorm {
    type Error = Error;

    fn try_from((idx, lm): (usize, &HashMap<String, TensorView<'_>>)) -> Result<Self> {
        Ok(Self {
            bias: tensor_to_array1(
                lm.get(&format!("ln{idx}.bias"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            ),
            weight: tensor_to_array1(
                lm.get(&format!("ln{idx}.weight"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            ),
        })
    }
}

impl TryFrom<&LM<'_>> for AttTime {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(AttTime {
            first: gk(lm, "att.time_first", tensor_to_array1)?,
            decay: (-gk(lm, "att.time_decay", tensor_to_array1)?.exp()).exp(),
            mix_k: Mix(gk(lm, "att.time_mix_k", tensor_to_array1)?),
            mix_v: Mix(gk(lm, "att.time_mix_v", tensor_to_array1)?),
            mix_r: Mix(gk(lm, "att.time_mix_r", tensor_to_array1)?),
        })
    }
}

impl TryFrom<&LM<'_>> for Attention {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(Attention {
            key_weight: gk(lm, "att.key.weight", tensor_to_array2)??,
            value_weight: gk(lm, "att.value.weight", tensor_to_array2)??,
            output_weight: gk(lm, "att.output.weight", tensor_to_array2)??,
            receptance_weight: gk(lm, "att.receptance.weight", tensor_to_array2)??,
            time: AttTime::try_from(lm)?,
        })
    }
}

impl TryFrom<&LM<'_>> for FFNTime {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FFNTime {
            mix_k: Mix(gk(lm, "ffn.time_mix_k", tensor_to_array1)?),
            mix_r: Mix(gk(lm, "ffn.time_mix_r", tensor_to_array1)?),
        })
    }
}

impl TryFrom<&LM<'_>> for FeedForwardNetwork {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: gk(lm, "ffn.key.weight", tensor_to_array2)??,
            value_weight: gk(lm, "ffn.value.weight", tensor_to_array2)??,
            receptance_weight: gk(lm, "ffn.receptance.weight", tensor_to_array2)??,
            time: FFNTime::try_from(lm)?,
        })
    }
}

impl TryFrom<&SafeTensors<'_>> for RWKV {
    type Error = Error;

    fn try_from(tensors: &SafeTensors<'_>) -> Result<Self> {
        let mut n_layers = 0;
        // This builds a HashMap of HashMaps.
        // The top level is None for non-layer tensors like "emb.weight" and
        // Some(layer_index) for each layer. The second level is just String key to TensorView.
        //
        // Worth noting is the fact that the model file gets mmaped but the actual keys/values
        // could be in any order. This means if you're loading from a spinny disky it could require
        // seeking all around the file rather than just reading sequentially.

        println!("* Discovering model structure.");
        let tm = tensors.tensors().into_iter().try_fold(
            HashMap::<Option<u32>, HashMap<String, TensorView>>::new(),
            |mut tm, (mut name, tensor)| {
                let (layer_num, ktv) = if let Some(rest) = name.strip_prefix("blocks.") {
                    let result = rest.split_once('.').ok_or_else(|| anyhow!("Bad format"))?;
                    let lnum = result.0.parse()?;
                    n_layers = n_layers.max(lnum + 1);
                    name = result.1.to_string();
                    (Some(lnum), tensor)
                } else {
                    (None, tensor)
                };

                tm.entry(layer_num)
                    .or_insert_with(Default::default)
                    .insert(name, ktv);
                Result::<_, Error>::Ok(tm)
            },
        )?;
        anyhow::ensure!(n_layers > 0, "Not even one measly layer?");

        let layers = (0..n_layers)
            .map(|lnum| {
                println!("-   Loading layer {}/{n_layers}", lnum + 1);
                let lm = tm.get(&Some(lnum)).expect("Impossible layer missing");
                Result::<_, Error>::Ok(Layer {
                    ln: [LayerNorm::try_from((1, lm))?, LayerNorm::try_from((2, lm))?],
                    att: Attention::try_from(lm)?,
                    ffn: FeedForwardNetwork::try_from(lm)?,
                })
                //
            })
            .collect::<Result<Vec<Layer>, _>>()?;
        let l0m = tm.get(&Some(0)).unwrap();
        let nlm = tm
            .get(&None)
            .ok_or_else(|| anyhow!("Missing non-layer tensors!"))?;
        println!("* Loading non-layer tensors.");
        Ok(RWKV {
            emb: gk(nlm, "emb.weight", tensor_to_array2)??,
            head: gk(nlm, "head.weight", tensor_to_array2)??,
            ln_out: LayerNorm {
                bias: gk(nlm, "ln_out.bias", tensor_to_array1)?,
                weight: gk(nlm, "ln_out.weight", tensor_to_array1)?,
            },
            ln0: LayerNorm::try_from((0, l0m))?,
            layers,
        })
    }
}
