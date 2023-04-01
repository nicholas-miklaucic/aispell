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
use arrayfire::{exp, max, row, sum_all, Array};
use lazy_static::lazy_static;
use tokenizers::Tokenizer;

use crate::{
    lm::LM,
    rwkv::{model::RWKVLayerState, util::pardot},
};

use self::{
    model::RWKV,
    util::{mmap_file, Elem},
};

/// The RWKV model state, including a loss.
#[derive(Clone)]
pub struct LossState {
    /// The previous tokens.
    pub tokens: Vec<u32>,
    /// The output logits for the next token. None if no prior state exists.
    pub logits: Option<Array<Elem>>,
    /// The model state.
    pub state: Vec<RWKVLayerState<Elem>>,
    /// The loss.
    pub loss: f64,
}

pub fn softmax(x: &Array<Elem>) -> Array<Elem> {
    let exp_x = exp(&(x - max(&x, 0)));
    let sum = sum_all(&exp_x).0;
    exp_x / sum
}

/// Like PyTorch item(), converts [x] to x. Panics if array is bigger than 1
/// element.
pub fn item(x: Array<Elem>) -> Elem {
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
    pub fn process_state(mut self, tokens: &[u32], rwkv: &RWKV<Elem>) -> Self {
        let [_n_vocab, _n_embed, _, _] = rwkv.emb.dims().get();
        let _n_layers = rwkv.layers.len();

        for token in tokens.iter() {
            self.loss += match &self.logits {
                Some(x) => item(row(&softmax(&x), *token as i64)).ln(),
                None => 0.0,
            } as f64;
            let initial_x = rwkv.ln0.norm(&row(&rwkv.emb, *token as i64));

            let x = rwkv
                .layers
                .iter()
                .enumerate()
                .fold(initial_x, |x, (lnum, layer)| {
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
    model: RWKV<Elem>,
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
        let dims = rwkv.emb.dims();
        let [_n_vocab, n_embed, _, _] = dims.get();
        let n_layers = rwkv.layers.len();

        let pretext = from_utf8(context).unwrap().to_owned();
        let text = from_utf8(completion).unwrap().to_owned();
        let pre_tokens = self.tokenizer.encode(pretext, true).unwrap();
        let tokens = self.tokenizer.encode(text, true).unwrap();

        let map = self.cache.lock().unwrap();

        let mut state = match map.get(pre_tokens.get_ids()) {
            Some(st) => st.clone(),
            None => {
                let st = LossState::new_empty(n_layers, *n_embed as usize);
                st.process_state(pre_tokens.get_ids(), rwkv)
            }
        };

        let mut curr_tokens = pre_tokens.get_ids().clone().to_vec();
        for token in tokens.get_ids() {
            curr_tokens.push(*token);
            state = match map.get(&curr_tokens) {
                Some(st) => st.clone(),
                None => state.clone().process_state(vec![*token].as_slice(), rwkv),
            };
        }

        state.loss
    }
}

lazy_static! {
    pub static ref RWKV_430M: Mutex<RwkvLM> = RwkvLM::try_new().unwrap().into();
}

#[cfg(test)]
mod tests {
    use arrayfire::{af_print, dim4, randu, Dim4};

    use super::*;

    #[test]
    fn test_helloworld() {
        dbg!("Hi");
        use arrayfire::*;

        info();
        dbg!("Info String:\n{}", info_string(true));
        println!("Arrayfire version: {:?}", get_version());
        let (name, platform, toolkit, compute) = device_info();
        dbg!(
            "Name: {}\nPlatform: {}\nToolkit: {}\nCompute: {}\n",
            name,
            platform,
            toolkit,
            compute
        );
        dbg!("Revision: {}", get_revision());

        let num_rows: i64 = 5;
        let num_cols: i64 = 3;
        let values: [f32; 3] = [1.0, 2.0, 3.0];
        let indices = Array::new(&values, Dim4::new(&[3, 1, 1, 1]));

        af_print!("Indices ", indices);

        let dims = Dim4::new(&[num_rows as u64, num_cols as u64, 1, 1]);

        let mut a = randu::<f32>(dims);
        af_print!("Create a 5-by-3 float   matrix on the GPU", a);

        println!("Element-wise arithmetic");
        let b = add(&sin(&a), &1.5f32, false);

        let b2 = add(&sin(&a), &cos(&a), false);

        let b3 = !&a;
        af_print!("sin(a) + 1.5 a.k.a b => ", b);
        af_print!("sin(a) + cos(a) => ", b2);
        af_print!("!a => ", b3);

        let test = a.clone() + b.clone();
        af_print!("a + b", test);

        let negation = -(a.clone());
        af_print!("-a ", negation);

        // Index array using sequences
        let seqs = &[Seq::new(1u32, 3, 1), Seq::default()];
        let sub = index(&a, seqs);
        af_print!("a(seq(1,3,1), span)", sub);

        //Index array using array and sequence
        let seq4gen = Seq::new(0u32, 2, 1);

        let mut idxrs = Indexer::default();
        idxrs.set_index(&indices, 0, None);
        idxrs.set_index(&seq4gen, 1, Some(false));

        let sub2 = index_gen(&a, idxrs);
        af_print!("a(indices, seq(0, 2, 1))", sub2);

        println!("Fourier transform the result");
        print(&fft(&b, 1.0, 0));

        println!("Grab last row & col of the random matrix");
        print(&a);
        print(&row(&a, num_rows - 1));
        print(&col(&a, num_cols - 1));

        let r_dims = Dim4::new(&[3, 1, 1, 1]);
        let r_input: [f32; 3] = [1.0, 1.0, 1.0];
        let r = Array::new(&r_input, r_dims);
        set_row(&mut a, &r, num_rows - 1);
        af_print!("Set last row to 1's", a);

        let d_dims = Dim4::new(&[2, 3, 1, 1]);
        let d_input: [i32; 6] = [1, 2, 3, 4, 5, 6];
        let d = Array::new(&d_input, d_dims);
        af_print!("Create 2-by-3 matrix from host data", d);

        //// // Sort A
        //println!("Sort A and print sorted array and corresponding indices");
        //let x = sort_index(&a, 0, true);
        //print(&x.0);
        //print(&x.1);

        let u8_cnst = &constant(1_u8, dims);
        af_print!("u8 constant array", u8_cnst);
        println!(
            "Is u8_cnst array float precision type ? {}",
            u8_cnst.is_single()
        );
    }

    #[test]
    fn test_af() {
        let num_rows: u64 = 5;
        let num_cols: u64 = 3;
        let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
        let a = randu::<f32>(dims);
        af_print!("Create a 5-by-3 matrix of random floats on the GPU", a);
        let v: Vec<f32> = vec![1.0, 2.0, 3.0];
        let arr = Array::new(&v, dim4!(3, 1, 1, 1));
        assert_eq!(arr.elements(), 3);
    }

    #[test]
    fn test_rwkv_state() {
        let lm = RWKV_430M.lock().unwrap();
        dbg!(lm.model.emb.dims());
    }

    #[test]
    fn test_prob_rwkv() {
        let ctxt = b"The cat is out of the ";
        let lm = RWKV_430M.lock().unwrap();
        dbg!("Hi!");
        let loss1 = lm.loss(ctxt, b"bag");
        let loss2 = lm.loss(ctxt, b"bug");
        assert!(loss1 < loss2, "\n{:.3} < {:.3} failed", loss1, loss2);
    }
}
