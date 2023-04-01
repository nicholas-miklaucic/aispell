use anyhow::{anyhow, Result};
use arrayfire::{dim4, print, Array, ConstGenerator, HasAfEnum, ImplicitPromote};
use mmap_rs::{MmapFlags, MmapOptions};
use num_traits::{One, Zero};
use safetensors::tensor::TensorView;

pub trait BaseReqOps
where
    Self: Sized
        + Default
        + Clone
        + HasAfEnum
        + Zero
        + One
        + ConstGenerator<OutType = Self>
        + ImplicitPromote<f64>,
{
}

impl<T> BaseReqOps for T where
    T: Sized
        + Default
        + Clone
        + HasAfEnum
        + Zero
        + One
        + ConstGenerator<OutType = T>
        + ImplicitPromote<f64>
{
}

pub trait ReqOps
where
    Self: BaseReqOps,
{
}

impl ReqOps for f32 {}
impl ReqOps for f64 {}

pub type Elem = f32;

pub trait ConvertBF16Tensor: ReqOps {
    fn tensor_to_array1(tensor: &TensorView<'_>) -> Array<Self>;
    fn tensor_to_array2(tensor: &TensorView<'_>) -> Result<Array<Self>>;
}

impl ConvertBF16Tensor for Elem {
    fn tensor_to_array1(tensor: &TensorView<'_>) -> Array<Self> {
        let elem_t = bf16_tensor_to_elem(tensor);
        let arr = Array::new(&elem_t, dim4!(elem_t.len() as u64, 1, 1, 1));
        arr
    }

    fn tensor_to_array2(tensor: &TensorView<'_>) -> Result<Array<Self>> {
        dbg!(tensor.shape());
        let shp = tensor
            .shape()
            .iter()
            .copied()
            .filter(|i| i != &1)
            .collect::<Vec<usize>>();
        anyhow::ensure!(shp.len() == 2, "Bad shape");
        Ok(Array::new(
            &bf16_tensor_to_elem(tensor),
            dim4!(shp[0] as u64, shp[1] as u64, 1, 1),
        ))
    }
}

/// Helper function for opening a file and mmaping it.
pub fn mmap_file(s: &str) -> Result<mmap_rs::Mmap> {
    let fp = std::fs::File::open(s)?;
    let flen = fp.metadata()?.len();
    unsafe {
        MmapOptions::new(flen as usize)
            .and_then(|mo| {
                mo.with_file(fp, 0)
                    .with_flags(MmapFlags::NO_CORE_DUMP)
                    .map()
            })
            .map_err(|e| anyhow!(e))
    }
}

/// Helper function to convert a SafeTensors TensorView into a flat
/// vector of Elem. The number of dimensions doesn't matter at this
/// point.
fn bf16_tensor_to_elem(tensor: &TensorView<'_>) -> Vec<Elem> {
    assert_eq!(tensor.dtype(), safetensors::Dtype::BF16);
    tensor
        .data()
        .chunks(2)
        .map(|i| half::bf16::from_le_bytes([i[0], i[1]]).to_f32())
        .collect()
}

/// Magical stuff I don't understand too well. It takes the list of probabilities
/// and chooses a reasonable tokenid based on that.

// disabling because I don't have rand

// pub fn sample_probs<T: ReqOps + num_traits::AsPrimitive<Elem>>(
//     rng: &mut impl rand::Rng,
//     probs: &ArrayView1<T>,
//     forever: bool, // Never select EndOfDocument token.
//     temperature: Elem,
//     top_p: Elem,
// ) -> usize {
//     use rand::distributions::{Distribution, WeightedError, WeightedIndex};

//     let mut sorted_probs = probs.as_slice().unwrap().to_vec();

//     sorted_probs.sort_by(|a, b| T::partial_cmp(a, b).unwrap().reverse());
//     let mut cumulative_probs = Vec::with_capacity(sorted_probs.len());
//     let _ = sorted_probs.iter().fold(T::zero(), |acc, i| {
//         let newcum = acc + *i;
//         cumulative_probs.push(newcum);
//         newcum
//     });
//     let cutoffidx = cumulative_probs
//         .iter()
//         .copied()
//         .enumerate()
//         .find(|(_, v)| v.as_() > top_p)
//         .map(|i| i.0)
//         .unwrap_or_default();
//     let cutoff = sorted_probs[cutoffidx].as_();
//     let mut probs = probs.map(|i| {
//         let i: Elem = i.as_();
//         if i < cutoff {
//             0.0
//         } else {
//             i
//         }
//     });
//     if forever {
//         probs[0] = 0.0;
//     }
//     let probs = &probs / probs.sum();
//     let dist = match WeightedIndex::new(probs.iter().map(|val| val.powf(1.0 / temperature))) {
//         Ok(dist) => dist,
//         Err(WeightedError::AllWeightsZero) => {
//             // Sorry if you wanted tokens forever, but this is how it's got to be.
//             return 0;
//         }
//         e => e.expect("I didn't sign up for this! (Bad weight in generated probability list.)"),
//     };
//     dist.sample(rng)
// }

#[allow(dead_code)]
mod dumdot {
    use arrayfire::{matmul, FloatingPoint, MatProp};

    use super::{Array, ReqOps};

    pub fn pardot<T: ReqOps + FloatingPoint>(lhs: &Array<T>, rhs: &Array<T>) -> Array<T> {
        matmul(lhs, rhs, MatProp::NONE, MatProp::NONE)
    }
}
pub use dumdot::pardot;
