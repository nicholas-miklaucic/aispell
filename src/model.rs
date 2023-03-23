//! `Model` describes the probability of specific errors occurring.

use std::f64::consts::PI;

use crate::edit::{edit_seqs, intermediate_edits, Edit};
use lazy_static::lazy_static;

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

/// Parameters describing of transposition likelihood using a logistic regression.
#[derive(Clone, Debug)]
pub struct TransposeLogitParams {
    /// The bias/intercept.
    pub bias: f64,

    /// The coefficient added when the transposed characters are on opposite
    /// sides of tke keyboard.
    pub two_hand_coef: f64,

    /// The coefficient of the column gap.
    pub col_gap_coef: f64,
}

impl TransposeLogitParams {
    /// The probability of transposing characters at the given positions..
    pub fn prob(&self, pos1: (f64, f64), pos2: (f64, f64), two_hands: bool) -> f64 {
        let (fx, fy) = pos1;
        let (tx, ty) = pos2;
        let (dx, _dy) = (tx - fx, ty - fy);

        let mut logit = self.bias;
        logit += if two_hands { self.two_hand_coef } else { 0.0 };
        logit += self.col_gap_coef * dx.abs();

        (1.0 + (-logit).exp()).recip()
    }
}

/// A model using a QWERTY-like keyboard layout. Documentation uses QWERTY for
/// examples.
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
    /// Parameters for the transposition probabilities.
    pub transpose_params: TransposeLogitParams,
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

    /// Returns whether the key would be typed with the right hand.
    pub fn is_right_hand(&self, key: u8) -> Option<bool> {
        let (x, _y) = self.key_posn(key)?;
        Some(2.0 * x >= self.row_len as f64)
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

    /// Gets replacement probability. Helper function that returns None instead of 0.
    fn try_replace_prob(&self, from: u8, to: u8) -> Option<f64> {
        let fpos = self.kbd.key_posn(from)?;
        let tpos = self.kbd.key_posn(to)?;

        let (fx, fy) = self.shift_pos(fpos);
        let (tx, ty) = self.shift_pos(tpos);
        let (dx, dy) = (tx - fx, ty - fy);

        // multivariate normal probablity distribution with no correlation
        // used to avoid having to import nalgebra

        let log_z = (PI * self.col_sd * self.row_sd * 2.0).ln();

        let distance = (dx / self.col_sd).powi(2) + (dy / self.row_sd).powi(2);

        let lprob = -0.5 * distance - log_z;
        Some((1.0 + (-lprob).exp()).recip())
    }

    /// Gets transposition probability. Helper function that returns None instead of 0.
    fn try_transpose_prob(&self, from: u8, to: u8) -> Option<f64> {
        let fpos = self.kbd.key_posn(from)?;
        let tpos = self.kbd.key_posn(to)?;

        let fhand = self.kbd.is_right_hand(from)?;
        let thand = self.kbd.is_right_hand(to)?;
        Some(
            self.transpose_params
                .prob(self.shift_pos(fpos), self.shift_pos(tpos), fhand ^ thand),
        )
    }

    /// Replacement probability.
    fn replace_prob(&self, from: u8, to: u8) -> f64 {
        self.try_replace_prob(from, to).unwrap_or(0.0)
    }

    /// Transposition probability.
    fn transpose_prob(&self, from: u8, to: u8) -> f64 {
        self.try_transpose_prob(from, to).unwrap_or(0.0)
    }
}

impl Default for TransposeLogitParams {
    fn default() -> Self {
        Self {
            bias: -3.8,
            two_hand_coef: 1.13,
            col_gap_coef: 0.88,
        }
    }
}

impl Default for KbdModel {
    fn default() -> Self {
        Self {
            kbd: QWERTY.clone(),
            key_offset: 0.05,
            col_sd: 8.13_f64.sqrt(),
            row_sd: 0.66_f64.sqrt(),
            transpose_params: TransposeLogitParams::default(),
        }
    }
}

impl Model<u8> for KbdModel {
    fn prob(&self, edit: &Edit, src: &[u8], tgt: &[u8]) -> f64 {
        let probability = match &edit {
            Edit::Insert(_, _) => 0.15,
            Edit::Delete(_) => 0.17,
            Edit::Transpose(i) => 0.15 * self.transpose_prob(src[*i], src[*i + 1]),
            Edit::Replace(i, j) => 0.52 * self.replace_prob(src[*i], tgt[*j]),
        };

        // a "catch-all" probability for other keys
        let other = 0.03;
        if probability.abs() < 1e-6 {
            other
        } else {
            (1.0 - other) * probability
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::from_utf8;

    use super::*;

    #[test]
    fn test_kbd_posn() {
        let kbd = QWERTY.clone();
        assert_eq!(kbd.key_posn(b'H').unwrap(), (5.0, 2.0));
        assert_eq!(kbd.key_posn(b'e').unwrap(), (2.0, 1.0));
    }

    #[test]
    fn test_kbd_hands() {
        let kbd = QWERTY.clone();
        assert!(kbd.is_right_hand(b'H').unwrap());
        assert!(!kbd.is_right_hand(b'e').unwrap());
    }

    #[test]
    fn test_probs() {
        let obs = "teh".as_bytes();
        let poss_corrs = vec![
            "tech".as_bytes(),
            "eh".as_bytes(),
            "the".as_bytes(),
            "ten".as_bytes(),
            "tea".as_bytes(),
            "troglodyte".as_bytes(),
        ];

        let model = KbdModel::default();

        let probs: Vec<f64> = poss_corrs
            .iter()
            .map(|src| model.total_prob(src, obs))
            .collect();

        for i in 0..probs.len() - 1 {
            dbg!((probs[i], probs[i + 1]));
            assert!(
                probs[i] >= probs[i + 1],
                "\nExpected {} before {}, but {:.4} < {:.4}\n{:?}",
                from_utf8(poss_corrs[i]).unwrap(),
                from_utf8(poss_corrs[i + 1]).unwrap(),
                probs[i],
                probs[i + 1],
                probs
            );
        }
    }
}
