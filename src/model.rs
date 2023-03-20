//! `Model` describes the probability of specific errors occurring.

use crate::edit::{edit_seqs, intermediate_edits, Edit};
use lazy_static::lazy_static;
use statrs::distribution::{Continuous, Normal};

/// An error model describing the probability of specific errors in sequences of
/// the given type.
pub trait Model<T: std::cmp::Eq + Clone> {
    /// The (unnormalized) probability of a specific error occurring.
    fn prob(&self, edit: &Edit, src: &[T], tgt: &[T]) -> f64;

    /// The (unnormalized) probability of the target sequence being produced
    /// from the first. Sums over all of the possible edit paths.
    fn total_prob(&self, src: &[T], tgt: &[T]) -> f64 {
        let seqs = edit_seqs(src, tgt);
        seqs.into_iter()
            .map(|seq| intermediate_edits(src, tgt, seq))
            .map(|seq| {
                seq.iter()
                    .map(|(curr, e)| self.prob(e, curr, tgt))
                    .product::<f64>()
            })
            .sum()
    }
}

/// A model using a QWERTY-like keyboard layout, assuming a Gaussian distribution
/// of substitutions. Documentation uses QWERTY for examples.
#[derive(Clone, Debug)]
pub struct KbdModel {
    /// The keyboard layout.
    pub kbd: Kbd,
    /// The tilt of the keyboard: 0.05 means that the A key is 0.05 key widths
    /// to the right of Q.
    pub key_offset: f64,
    /// The standard deviation of errors in the x-axis, like typing "R" for "T".
    /// In units of the width between two keys.
    pub col_sd: f64,
    /// The standard deviation of errors in the y-axis, like typing "G" for "T".
    /// In units of the width between two keys.
    pub row_sd: f64,
}

/// A keyboard layout.
#[derive(Clone, Debug)]
pub struct Kbd {
    /// The keys from left to right and bottom to top: z, x, c, ..., a, s, ...
    pub keys: Vec<u8>,
    /// The keys when Shift is pressed, of the same shape as keys.
    pub shift_keys: Vec<u8>,
    /// The number of keys per row: 10 for QWERTY.
    pub row_len: usize,
}

impl Kbd {
    /// Returns the position of the key in a coordinate system where (0, 0) is Z
    /// and 1 is a single key width. `None` is returned if the key is not on the
    /// keyboard.
    pub fn key_posn(&self, key: u8) -> Option<(f64, f64)> {
        let key_i = self.keys.iter().position(|&x| x == key);
        let skey_i = self.shift_keys.iter().position(|&x| x == key);
        key_i.or(skey_i).map(|k| {
            let x = (k as f64) % (self.row_len as f64);
            let y = (k as f64).div_euclid(self.row_len as f64);
            (x, y)
        })
    }
}

lazy_static! {
    static ref QWERTY: Kbd = Kbd {
        keys: "1234567890qwertyuiopasdfghjkl;zxcvbnm,./".as_bytes().into(),
        shift_keys: "!@#$%^&*()QWERTYUIOPASDFGHJKL:ZXCVBNM<>?".as_bytes().into(),
        row_len: 10,
    };
}

impl KbdModel {
    /// Adjusts the position to account for the offset.
    fn shift_pos(&self, pos: (f64, f64)) -> (f64, f64) {
        let (x, y) = pos;
        (x - self.key_offset * y, y)
    }

    /// Helper function that returns None instead of 0.
    fn try_replace_prob(&self, from: u8, to: u8) -> Option<f64> {
        let fpos = self.kbd.key_posn(from)?;
        let tpos = self.kbd.key_posn(to)?;

        let (fx, fy) = self.shift_pos(fpos);
        let (tx, ty) = self.shift_pos(tpos);
        let (dx, dy) = (tx - fx, ty - fy);

        Some(
            Normal::new(0.0, self.col_sd).unwrap().pdf(dx)
                * Normal::new(0.0, self.row_sd).unwrap().pdf(dy),
        )
    }

    /// Normalized probability of replacement occurring (normalized across all replacements)
    fn replace_prob(&self, from: u8, to: u8) -> f64 {
        self.try_replace_prob(from, to).unwrap_or(0.0)
    }
}

impl Default for KbdModel {
    fn default() -> Self {
        Self {
            kbd: QWERTY.clone(),
            key_offset: 0.04,
            col_sd: 0.8,
            row_sd: 0.14,
        }
    }
}

impl Model<u8> for KbdModel {
    fn prob(&self, edit: &Edit, src: &[u8], tgt: &[u8]) -> f64 {
        match &edit {
            // TODO: find relative weights
            // also replace transpose, insert, delete with model
            Edit::Insert(_, _) => 0.1,
            Edit::Delete(_) => 0.1,
            Edit::Transpose(_) => 0.6,
            Edit::Replace(i, j) => 0.2 * (self.replace_prob(src[*i], tgt[*j]) + 0.05),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probs() {
        let obs = "teh".as_bytes();
        let poss_corrs = vec![
            "the".as_bytes(),
            "ten".as_bytes(),
            "tea".as_bytes(),
            "eh".as_bytes(),
            "tech".as_bytes(),
        ];

        let model = KbdModel::default();

        let probs: Vec<f64> = poss_corrs
            .iter()
            .map(|src| model.total_prob(src, obs))
            .collect();

        for i in 0..probs.len() - 1 {
            dbg!((probs[i], probs[i + 1]));
            assert!(probs[i] >= probs[i + 1]);
        }
    }
}
