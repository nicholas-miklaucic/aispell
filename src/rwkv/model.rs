#![allow(clippy::upper_case_acronyms)]
use arrayfire::{
    clamp, constant, exp, max_all, meanvar, pow, row, sigmoid, sqrt, sum_all, Array, VarianceBias,
};

use super::util::{pardot, Elem, ReqOps};

#[derive(Clone)]
/// Corresponds to:
/// 1. blocks.N.att.time_mix_[kvr]
/// 2. blocks.N.ffn.time_mix_[kr]
pub struct Mix<T: ReqOps>(pub Array<T>);

#[derive(Clone)]
/// Corresponds to:
/// 1. ln_out.[bias,weight]
/// 2. blocks.N.ln[012].[bias,weight]
/// However, note that ln0 only exists in block 0.
pub struct LayerNorm<T: ReqOps> {
    pub bias: Array<T>,
    pub weight: Array<T>,
}

#[derive(Clone)]
/// Corresponds to:
/// 1. blocks.N.time_[first,decay]
/// 2. blocks.N.time_mix_[kvr]
pub struct AttTime<T: ReqOps> {
    pub decay: Array<T>,
    pub mix_k: Mix<T>,
    pub mix_v: Mix<T>,
    pub mix_r: Mix<T>,
    pub first: Array<T>,
}

/// Corresponds to:
/// 1. blocks.N.ffn.time_mix_[kr]
#[derive(Clone)]
pub struct FFNTime<T: ReqOps> {
    pub mix_k: Mix<T>,
    pub mix_r: Mix<T>,
}

/// Corresponds to:
/// 1. blocks.N.att.[key,value,output,receptance].weight
/// 3. Keys described in AttTime.
#[derive(Clone)]
pub struct Attention<T: ReqOps> {
    pub key_weight: Array<T>,
    pub value_weight: Array<T>,
    pub output_weight: Array<T>,
    pub receptance_weight: Array<T>,
    pub time: AttTime<T>,
}

/// Corresponds to:
/// 1. blocks.N.ffn.[key,value,receptance].weight
/// 3. Keys described in FFNTime.
#[derive(Clone)]
pub struct FeedForwardNetwork<T: ReqOps> {
    pub key_weight: Array<T>,
    pub value_weight: Array<T>,
    pub receptance_weight: Array<T>,
    pub time: FFNTime<T>,
}

#[derive(Clone)]
/// See the comments for Attention, FeedForwardNetwork and LayerNorm.
/// Note though that the array of LayerNorm here corresponds with
/// blocks.N.ln[12] so array index 0 is ln1.
pub struct Layer<T: ReqOps> {
    /// ln[0] (AKA ln1) is used for time mixing,
    /// ln[1] (AKA ln2) is used for channel mixing.
    pub ln: [LayerNorm<T>; 2],
    pub att: Attention<T>,
    pub ffn: FeedForwardNetwork<T>,
}

#[derive(Clone)]
pub struct RWKV<T: ReqOps> {
    /// emb.weight
    pub emb: Array<T>,
    /// head.weight
    pub head: Array<T>,
    /// ln_out.[weight,bias]
    pub ln_out: LayerNorm<T>,
    /// This is actually blocks.0.ln0
    pub ln0: LayerNorm<T>,
    pub layers: Vec<Layer<T>>,
}

#[derive(Clone)]
/// Each layer has its own independent state.
pub struct RWKVLayerState<T: ReqOps> {
    /// State from time mixing.
    pub tm_state: Array<T>,
    /// Time mixing numerator?
    pub tm_num: Array<T>,
    /// Time mixing denominator?
    pub tm_den: Array<T>,
    /// State from channel mixing.
    pub cm_state: Array<T>,
}

impl<T: ReqOps> RWKVLayerState<T> {
    pub fn new(n_embed: usize) -> Self {
        let zs = constant!(T::zero(); n_embed as u64);
        Self {
            tm_state: zs.clone(),
            tm_num: zs.clone(),
            tm_den: zs.clone(),
            cm_state: zs,
        }
    }

    /// Updates the state for this layer.
    pub fn update(
        &mut self,
        tm_state: Array<T>,
        tm_num: Array<T>,
        tm_den: Array<T>,
        cm_state: Array<T>,
    ) {
        *self = Self {
            tm_state,
            tm_num,
            tm_den,
            cm_state,
        }
    }
}

impl Mix<Elem> {
    pub fn mix(&self, x: &Array<Elem>, last_x: &Array<Elem>) -> Array<Elem> {
        let ans: Array<Elem> = x * &self.0 + last_x * (1.0_f32 - &self.0);
        ans
    }
}

impl Attention<Elem> {
    pub fn time_mixing(
        &self,
        x: &Array<Elem>,
        state: &RWKVLayerState<Elem>,
    ) -> (Array<Elem>, (Array<Elem>, Array<Elem>)) {
        let last_x = &state.tm_state;
        let last_num = &state.tm_num;
        let last_den = &state.tm_den;

        let k = pardot(&self.key_weight, &self.time.mix_k.mix(x, last_x));
        let v = pardot(&self.value_weight, &self.time.mix_v.mix(x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(x, last_x));

        let exp_k = exp(&k);
        let exp_decay = exp(&-exp(&self.time.decay));

        let wkv = {
            let e = exp(&(&self.time.first + &k));
            (last_num + &e * &v) / (last_den + e)
        };
        let rwkv = sigmoid(&r) * wkv;

        let num = &exp_decay * last_num + &exp_k * &v;
        let den = &exp_decay * last_den + &exp_k;
        (pardot(&self.output_weight, &rwkv), (num, den))
    }
}

impl FeedForwardNetwork<Elem> {
    pub fn channel_mixing(&self, x: &Array<Elem>, state: &RWKVLayerState<Elem>) -> Array<Elem> {
        let last_x = &state.cm_state;
        let k = pardot(&self.key_weight, &self.time.mix_k.mix(x, last_x));
        let r = pardot(&self.receptance_weight, &self.time.mix_r.mix(x, last_x));
        let vk_vec: Array<Elem> = pow(&clamp(&k, &std::f32::MIN, &2.0, true), &2.0, true).cast();
        let vk = pardot(&self.value_weight, &vk_vec);

        sigmoid(&r) * &vk
    }
}

impl LayerNorm<Elem> {
    /// Normalize a 1D array.
    pub fn norm(&self, x: &Array<Elem>) -> Array<Elem> {
        let (mean, var) = meanvar(
            x,
            &constant!(1.0; x.elements() as u64),
            VarianceBias::DEFAULT,
            0,
        );
        let std = sqrt(&var);
        (x - mean) / std * &self.weight + &self.bias
    }
}

impl RWKV<Elem> {
    /// Evaluates a layer. Each layer must be evaluated in sequence,
    /// serially as they each generate "x" and also require "x" as input.
    pub fn evaluate_layer(
        &self,
        mut x: Array<Elem>,
        layer: &Layer<Elem>,
        layer_state: &mut RWKVLayerState<Elem>,
    ) -> Array<Elem> {
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
    pub fn evaluate(&self, token: usize, state: &mut [RWKVLayerState<Elem>]) -> Array<Elem> {
        let emb = &self.emb;
        let initial_x = self.ln0.norm(&row(emb, token as i64));

        let x = self
            .layers
            .iter()
            .enumerate()
            .fold(initial_x, |x, (lnum, layer)| {
                self.evaluate_layer(x, layer, &mut state[lnum])
            });

        let x = pardot(&self.head, &self.ln_out.norm(&x));
        let x_max = max_all(&x).0;
        let e_x = exp(&(x - x_max));

        &e_x / sum_all(&e_x).0
    }
}
