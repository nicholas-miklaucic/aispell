use fast_symspell::{AsciiStringStrategy, Suggestion, SymSpell, SymSpellBuilder, Verbosity};
use model::Model;
use seqalign::measures::LevenshteinDamerau;
use seqalign::Align;

// // Get the edit script.
use seqalign::measures::LevenshteinDamerauOp;
use seqalign::op::IndexedOperation;

pub mod edit;
pub mod model;

fn check_suggestions(word: &'static str, edit_distance: i64) -> Vec<Suggestion> {
    let mut symspell: SymSpell<AsciiStringStrategy> = SymSpellBuilder::default()
        .max_dictionary_edit_distance(edit_distance)
        .prefix_length(2)
        .build()
        .unwrap();

    symspell.load_dictionary("data/frequency_dictionary_en_82_765.txt", 0, 1, " ");

    symspell.lookup(word, Verbosity::All, edit_distance)
}

/// A correction with a probability for a word.
#[derive(Debug, Clone)]
pub struct Correction {
    /// The suggested correction.
    pub term: String,

    /// The probability of the term being correct.
    pub prob: f64,
}

/// Returns a list of corrections, sorted by most probable to least.
fn corrections<T: Model<u8>>(word: &'static str, edit_distance: i64, model: &T) -> Vec<Correction> {
    let mut corrs: Vec<Correction> = check_suggestions(word, edit_distance)
        .into_iter()
        .map(|sug| Correction {
            term: sug.term.clone(),
            prob: model.total_prob(word.as_bytes(), sug.term.as_bytes()),
        })
        .collect();

    // note how b is first, not a: this sorts in descending order
    corrs.sort_unstable_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    corrs
}

fn edit_ops(orig: &'static str, sug: Suggestion) -> Vec<IndexedOperation<LevenshteinDamerauOp>> {
    let src = sug.term.as_bytes();
    let tgt = orig.as_bytes();

    let measure = LevenshteinDamerau::new(1, 1, 1, 1);
    let alignment = measure.align(src, tgt);

    alignment.edit_script()
}

#[cfg(test)]
mod tests {
    use crate::model::KbdModel;

    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn it_works() {
        let orig = "otol";
        let result = check_suggestions(orig, 2);
        //assert_eq!(result, vec![]);

        // let comps: Vec<String> = result
        //     .iter()
        //     .take(3)
        //     .map(|s| s.term.clone().into())
        //     .collect();
        // assert_eq!(comps, vec!["".to_string()]);

        let ops: Vec<Vec<IndexedOperation<LevenshteinDamerauOp>>> = result
            .into_iter()
            .take(3)
            .map(|s| edit_ops(orig, s))
            .collect();

        // let test_ops: Vec<Vec<IndexedOperation<LevenshteinDamerauOp>>> = vec![];
        // assert_eq!(ops, test_ops);
    }

    #[test]
    fn test_corrections() {
        let orig = "otol";
        let corrs = corrections(orig, 1, &KbdModel::default());
        assert_eq!(
            corrs.into_iter().map(|c| c.term).collect::<Vec<String>>(),
            vec!["tool".to_string()]
        );
    }

    #[test]
    fn test_corrections1() {
        let orig = "teh";
        let corrs = corrections(orig, 1, &KbdModel::default());
        let probs: Vec<f64> = corrs.iter().map(|c| c.prob).collect();
        let terms: Vec<String> = corrs.into_iter().map(|c| c.term).collect();
        assert_eq!(terms, vec!["artificial".to_string()]);
    }
}
