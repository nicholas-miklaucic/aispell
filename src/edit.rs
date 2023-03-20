//! Computes the optimal edit sequence to transform one string into another.

use std::iter::repeat;

use seqalign::{
    measures::{LevenshteinDamerau, LevenshteinDamerauOp},
    op::IndexedOperation,
    Align,
};

/// A single Damerau-Levenshtein operation: transpose, insert, delete,
/// substitute. Values are (src index, target index)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Edit {
    /// Insertion such that the new value at the src index matches the target index.
    Insert(usize, usize),

    /// Deletion of the character at the given index of src.
    Delete(usize),

    /// Transposition of the character at the index with the following one.
    Transpose(usize),

    /// Replacement of the character at the index in src with the one at the index in target.
    Replace(usize, usize),
}

impl Edit {
    /// Gets the new src after applying the edit.
    pub fn edited<T: std::cmp::Eq + Clone>(&self, src: &[T], tgt: &[T]) -> Vec<T> {
        let mut new_src = src.to_vec();
        match &self {
            Edit::Insert(i, j) => new_src.insert(*i, tgt[*j].clone()),
            Edit::Delete(i) => {
                new_src.remove(*i);
            }
            Edit::Transpose(i) => {
                new_src.swap(*i, *i + 1);
            }
            Edit::Replace(i, j) => {
                new_src[*i] = tgt[*j].clone();
            }
        }
        new_src
    }
}

impl TryFrom<IndexedOperation<LevenshteinDamerauOp>> for Edit {
    type Error = ();

    fn try_from(value: IndexedOperation<LevenshteinDamerauOp>) -> Result<Self, Self::Error> {
        let src = value.source_idx();
        let tgt = value.target_idx();
        match value.operation() {
            LevenshteinDamerauOp::Insert(_) => Ok(Self::Insert(src, tgt)),
            LevenshteinDamerauOp::Delete(_) => Ok(Self::Delete(src)),
            LevenshteinDamerauOp::Match => Err(()),
            LevenshteinDamerauOp::Substitute(_) => Ok(Self::Replace(src, tgt)),
            LevenshteinDamerauOp::Transpose(_) => Ok(Self::Transpose(src)),
        }
    }
}

/// Converts the operations to `Edit`s, adjusting indices to match the intermediate results.
pub fn convert_to_edits(ops: Vec<IndexedOperation<LevenshteinDamerauOp>>) -> Vec<Edit> {
    let mut raw_edits: Vec<Edit> = ops.into_iter().filter_map(|o| o.try_into().ok()).collect();
    fix_edit_indexing(&mut raw_edits);
    raw_edits
}

/// Adjusts edit indices to match the intermediate steps.
pub fn fix_edit_indexing(raw_edits: &mut Vec<Edit>) {
    for i in 0..raw_edits.len() {
        match raw_edits[i] {
            Edit::Insert(src_i, _) => {
                let fix_ind = |a| if a >= src_i { a + 1 } else { a };
                for j in (i + 1)..raw_edits.len() {
                    raw_edits[j] = match raw_edits[j] {
                        Edit::Insert(a, b) => Edit::Insert(fix_ind(a), b),
                        Edit::Delete(a) => Edit::Delete(fix_ind(a)),
                        Edit::Transpose(a) => Edit::Transpose(fix_ind(a)),
                        Edit::Replace(a, b) => Edit::Replace(fix_ind(a), b),
                    }
                }
            }
            Edit::Delete(src_i) => {
                let fix_ind = |a| if a > src_i { a - 1 } else { a };
                for j in (i + 1)..raw_edits.len() {
                    raw_edits[j] = match raw_edits[j] {
                        Edit::Insert(a, b) => Edit::Insert(fix_ind(a), b),
                        Edit::Delete(a) => Edit::Delete(fix_ind(a)),
                        Edit::Transpose(a) => Edit::Transpose(fix_ind(a)),
                        Edit::Replace(a, b) => Edit::Replace(fix_ind(a), b),
                    }
                }
            }
            // transpose/replace don't affect indexing
            _ => {}
        }
    }
}

/// Gets the edit sequences between two strings
pub fn edit_seqs<T: std::cmp::Eq>(src: &[T], tgt: &[T]) -> Vec<Vec<Edit>> {
    let measure = LevenshteinDamerau::new(1, 1, 1, 1);
    let alignment = measure.align(src, tgt);

    alignment
        .edit_scripts()
        .into_iter()
        .map(convert_to_edits)
        .collect()
}

/// Takes an edit sequence and returns the list of intermediate values.
pub fn intermediate_edits<T>(src: &[T], tgt: &[T], seq: Vec<Edit>) -> Vec<(Vec<T>, Edit)>
where
    T: std::cmp::Eq + Clone,
{
    seq.into_iter()
        .scan(src.to_vec(), |curr, e| {
            *curr = e.edited(curr, tgt);
            Some((curr.to_vec(), e))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::str::from_utf8;

    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_indexing() {
        let src = "otol".as_bytes();
        let tgt = "told".as_bytes();
        let edits = &edit_seqs(src, tgt)[0];
        let intermediates: Vec<String> = intermediate_edits(src, tgt, dbg!(edits.to_vec()))
            .into_iter()
            .map(|(i, _e)| from_utf8(&i).unwrap().to_string())
            .collect();
        assert_eq!(intermediates, vec!["tol", "told"]);
    }

    #[test]
    fn test_edits() {
        let edits = &edit_seqs("Byaes".as_bytes(), "Bayesian".as_bytes());
        assert_eq!(edits.len(), 1);

        assert_eq!(
            edits[0].to_vec(),
            vec![
                Edit::Transpose(1),
                Edit::Insert(5, 5),
                Edit::Insert(6, 6),
                Edit::Insert(7, 7)
            ]
        );

        // Byaes
        // Bayes
        // Bayesn
        // Bayesan
        // Bayesian
    }

    #[test]
    fn test_insert() {
        let src = "elan".as_bytes();
        let tgt = "eland".as_bytes();
        let edits = &edit_seqs(src, tgt)[0];
        assert_eq!(edits, &vec![Edit::Insert(4, 4)]);
    }

    #[test]
    fn test_delete_insert() {
        let src = "helan".as_bytes();
        let tgt = "eland".as_bytes();
        let edits = &edit_seqs(src, tgt)[0];
        assert_eq!(edits, &vec![Edit::Delete(0), Edit::Insert(4, 4)]);
    }

    #[test]
    fn test_fix_indexing() {
        let mut edits = vec![Edit::Delete(0), Edit::Insert(5, 4)];
        fix_edit_indexing(&mut edits);
        assert_eq!(edits, vec![Edit::Delete(0), Edit::Insert(4, 4)]);
    }

    #[test]
    fn test_intermediate_edits() {
        let words = vec!["person", "woman", "man", "camera", "tv", "velvet", "heart"];

        for w1 in words.iter() {
            for w2 in words.iter() {
                if w1 != w2 {
                    let src = w1.as_bytes();
                    let tgt = w2.as_bytes();
                    for seq in edit_seqs(src, tgt) {
                        let ints = intermediate_edits(src, tgt, seq.clone());
                        let (ult, _e) = &ints[ints.len() - 1];
                        assert_eq!(
                            ult,
                            tgt,
                            "\nDid not match: {:?}, {} → {}\n{:?}\n{:?}\n{:?}",
                            from_utf8(&ult),
                            w1,
                            w2,
                            seq,
                            ints.iter().map(|(_s, e)| e).collect::<Vec<&Edit>>(),
                            ints.iter()
                                .map(|(s, _e)| from_utf8(s).unwrap())
                                .collect::<Vec<&str>>()
                        );
                    }
                }
            }
        }
    }
}
