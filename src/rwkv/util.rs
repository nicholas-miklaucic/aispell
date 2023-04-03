use anyhow::{anyhow, Result};
use mmap_rs::{MmapFlags, MmapOptions};
use safetensors::tensor::TensorView;
use tch::{Kind, Tensor};

pub const ELEM: tch::Kind = Kind::BFloat16;

pub fn tensor_to_array1(tensor: &TensorView<'_>) -> Tensor {
    let elem_t = bf16_tensor_to_elem(tensor);
    let arr = Tensor::of_slice(&elem_t);
    arr
}

pub fn tensor_to_array2(tensor: &TensorView<'_>) -> Result<Tensor> {
    let shp = tensor
        .shape()
        .iter()
        .copied()
        .filter(|i| i != &1)
        .collect::<Vec<usize>>();
    anyhow::ensure!(shp.len() == 2, "Bad shape");
    let elem_t = bf16_tensor_to_elem(tensor);
    Ok(Tensor::of_slice(&elem_t).reshape(&[shp[0] as i64, shp[1] as i64]))
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
fn bf16_tensor_to_elem(tensor: &TensorView<'_>) -> Vec<f32> {
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
    use super::Tensor;

    pub fn pardot(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        lhs.matmul(rhs)
    }
}
pub use dumdot::pardot;
