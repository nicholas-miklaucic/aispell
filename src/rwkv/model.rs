#![allow(clippy::upper_case_acronyms)]
use tch::{Device, IndexOp, Kind, Tensor};

use super::util::{pardot, ELEM};

/// Corresponds to:
/// 1. blocks.N.att.time_mix_[kvr]
/// 2. blocks.N.ffn.time_mix_[kr]
pub struct Mix(pub Tensor);

impl Clone for Mix {
    fn clone(&self) -> Self {
        Self(self.0.copy())
    }
}

/// Corresponds to:
/// 1. ln_out.[bias,weight]
/// 2. blocks.N.ln[012].[bias,weight]
/// However, note that ln0 only exists in block 0.
pub struct LayerNorm {
    pub bias: Tensor,
    pub weight: Tensor,
}

impl Clone for LayerNorm {
    fn clone(&self) -> Self {
        Self {
            bias: self.bias.copy(),
            weight: self.weight.copy(),
        }
    }
}

/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
pub struct AttTime {
    pub decay: Tensor,
    pub mix_k: Mix,
    pub mix_v: Mix,
    pub mix_r: Mix,
    pub first: Tensor,
}

impl Clone for AttTime {
    fn clone(&self) -> Self {
        Self {
            decay: self.decay.copy(),
            mix_k: self.mix_k.clone(),
            mix_v: self.mix_v.clone(),
            mix_r: self.mix_r.clone(),
            first: self.first.copy(),
        }
    }
}

/// Corresponds to:
/// 1. blocks.N.ffn.time_mix_[kr]
#[derive(Clone)]
pub struct FFNTime {
    pub mix_k: Mix,
    pub mix_r: Mix,
}

/// Corresponds to:
/// 1. blocks.N.att.[key,value,output,receptance].weight
/// 3. Keys described in AttTime.
pub struct Attention {
    pub key_weight: Tensor,
    pub value_weight: Tensor,
    pub output_weight: Tensor,
    pub receptance_weight: Tensor,
    pub time: AttTime,
}

impl Clone for Attention {
    fn clone(&self) -> Self {
        Self {
            key_weight: self.key_weight.copy(),
            value_weight: self.value_weight.copy(),
            output_weight: self.output_weight.copy(),
            receptance_weight: self.receptance_weight.copy(),
            time: self.time.clone(),
        }
    }
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
pub struct FeedForwardNetwork {
    pub key_weight: Tensor,
    pub value_weight: Tensor,
    pub receptance_weight: Tensor,
    pub time: FFNTime,
}

impl Clone for FeedForwardNetwork {
    fn clone(&self) -> Self {
        Self {
            key_weight: self.key_weight.copy(),
            value_weight: self.value_weight.copy(),
            receptance_weight: self.receptance_weight.copy(),
            time: self.time.clone(),
        }
    }
}

/// See the comments for Attention, FeedForwardNetwork and LayerNorm.
/// Note though that the array of LayerNorm here corresponds with
/// blocks.N.ln[12] so array index 0 is ln1.
#[derive(Clone)]
pub struct Layer {
    /// ln[0] (AKA ln1) is used for time mixing,
    /// ln[1] (AKA ln2) is used for channel mixing.
    pub ln: [LayerNorm; 2],
    pub att: Attention,
    pub ffn: FeedForwardNetwork,
}

pub struct RWKV {
    /// emb.weight
    pub emb: Tensor,
    /// head.weight
    pub head: Tensor,
    /// ln_out.[weight,bias]
    pub ln_out: LayerNorm,
    /// This is actually blocks.0.ln0
    pub ln0: LayerNorm,
    pub layers: Vec<Layer>,
}

impl Clone for RWKV {
    fn clone(&self) -> Self {
        Self {
            emb: self.emb.copy(),
            head: self.head.copy(),
            ln_out: self.ln_out.clone(),
            ln0: self.ln0.clone(),
            layers: self.layers.clone(),
        }
    }
}

/// Each layer has its own independent state.
pub struct RWKVLayerState {
    /// State from time mixing.
    pub tm_state: Tensor,
    /// Time mixing numerator?
    pub tm_num: Tensor,
    /// Time mixing denominator?
    pub tm_den: Tensor,
    /// State from channel mixing.
    pub cm_state: Tensor,
}

impl Clone for RWKVLayerState {
    fn clone(&self) -> Self {
        Self {
            tm_state: self.tm_state.copy(),
            tm_num: self.tm_num.copy(),
            tm_den: self.tm_den.copy(),
            cm_state: self.cm_state.copy(),
        }
    }
}

impl RWKVLayerState {
    pub fn new(n_embed: usize) -> Self {
        let zs = Tensor::zeros(&[n_embed as i64], (ELEM, Device::Cpu));
        Self {
            tm_state: zs.copy(),
            tm_num: zs.copy(),
            tm_den: zs.copy(),
            cm_state: zs,
        }
    }

    /// Updates the state for this layer.
    pub fn update(&mut self, tm_state: Tensor, tm_num: Tensor, tm_den: Tensor, cm_state: Tensor) {
        *self = Self {
            tm_state,
            tm_num,
            tm_den,
            cm_state,
        }
    }
}

impl Mix {
    pub fn mix(&self, x: &Tensor, last_x: &Tensor) -> Tensor {
        // dbg!(x.size(), last_x.size(), self.0.size());
        x * &self.0 + last_x * (1.0_f32 - &self.0)
    }
}

impl Attention {
    pub fn time_mixing(&self, x: &Tensor, state: &RWKVLayerState) -> (Tensor, (Tensor, Tensor)) {
        let last_x = &state.tm_state;
        let last_num = &state.tm_num;
        let last_den = &state.tm_den;

        let k = pardot(&self.key_weight, &self.time.mix_k.mix(&x, last_x));
        let v = pardot(&self.value_weight, &self.time.mix_v.mix(&x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(&x, last_x));

        let exp_k = k.exp();
        let exp_decay = &self.time.decay;

        let wkv = {
            let e = (&self.time.first + k).exp();
            (last_num + &e * &v) / (last_den + e)
        };
        let rwkv = r.sigmoid() * wkv;

        let num = exp_decay * last_num + &exp_k * &v;
        let den = exp_decay * last_den + &exp_k;
        (pardot(&self.output_weight, &rwkv), (num, den))
    }
}

impl FeedForwardNetwork {
    pub fn channel_mixing(&self, x: &Tensor, state: &RWKVLayerState) -> Tensor {
        let last_x = &state.cm_state;
        let k = pardot(&self.key_weight, &self.time.mix_k.mix(x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(x, last_x));
        let vk_vec = k.relu().square();
        let vk = pardot(&self.value_weight, &vk_vec);

        r.sigmoid() * &vk
    }
}

impl LayerNorm {
    /// Normalize a 1D array.
    pub fn norm(&self, x: &Tensor) -> Tensor {
        let (std, mean) = x.std_mean_dim(vec![-1_i64].as_slice(), false, true);
        (x - mean) / std * &self.weight + &self.bias
    }
}

impl RWKV {
    /// Evaluates a layer. Each layer must be evaluated in sequence,
    /// serially as they each generate "x" and also require "x" as input.
    pub fn evaluate_layer(
        &self,
        mut x: Tensor,
        layer: &Layer,
        layer_state: &mut RWKVLayerState,
    ) -> Tensor {
        let x_ln1 = layer.ln[0].norm(&x);
        let (dx, (tm_num, tm_den)) = layer.att.time_mixing(&x_ln1, layer_state);
        x += dx;

        let x_ln2 = layer.ln[1].norm(&x);
        let dx = layer.ffn.channel_mixing(&x_ln2, layer_state);
        x += dx;

        layer_state.update(x_ln1, tm_num, tm_den, x_ln2);
        x
    }

    /// Evaluate all the layers and return a list of probabilities.
    pub fn evaluate(&self, token: usize, state: &mut [RWKVLayerState]) -> Tensor {
        let emb = &self.emb;
        let initial_x = self.ln0.norm(&emb.i(token as i64));

        let x = self
            .layers
            .iter()
            .enumerate()
            .fold(initial_x, |x, (lnum, layer)| {
                self.evaluate_layer(x, layer, &mut state[lnum])
            });

        let x = pardot(&self.head, &self.ln_out.norm(&x));
        x.softmax(-1, Kind::Float)
    }
}
