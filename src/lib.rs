use fast_symspell::{AsciiStringStrategy, Suggestion, SymSpell, SymSpellBuilder, Verbosity};
use lm::LM;
use model::Model;
use tch::{Kind, Tensor};

pub mod edit;
pub mod lm;
pub mod model;

fn check_suggestions(word: &'static str, edit_distance: i64) -> Vec<Suggestion> {
    let mut symspell: SymSpell<AsciiStringStrategy> = SymSpellBuilder::default()
        .max_dictionary_edit_distance(edit_distance)
        .prefix_length(2)
        .count_threshold(0)
        .build()
        .unwrap();

    symspell.load_dictionary("data/custom_dictionary.txt", 0, 1, " ");

    symspell.lookup(word, Verbosity::All, edit_distance)
}

/// A correction with a probability for a word.
#[derive(Debug, Clone)]
pub struct Correction {
    /// The suggested correction.
    pub term: String,

    /// The probability of the term being correct.
    pub prob: f64,

    /// The keyboard probability.
    kbd_prob: f64,

    /// The language model probability.
    lm_prob: f64,
}

/// Returns a list of corrections, sorted by most probable to least.
pub fn corrections<T: Model<u8>, L: LM<u8>>(
    word: &'static str,
    edit_distance: i64,
    kbd_model: &T,
    context: Option<&'static str>,
    lang_model: &L,
) -> Vec<Correction> {
    let mut corrs: Vec<Correction> = check_suggestions(word, edit_distance)
        .into_iter()
        .map(|sug| Correction {
            term: sug.term.clone(),
            prob: 0.0,
            kbd_prob: kbd_model.total_prob(word.as_bytes(), sug.term.as_bytes()),
            lm_prob: 0.0,
        })
        .collect();

    let losses = Tensor::of_slice(
        &corrs
            .iter()
            .map(|corr| lang_model.loss(context.unwrap_or(" ").as_bytes(), corr.term.as_bytes()))
            .collect::<Vec<_>>(),
    );
    let losses = (losses * -3).softmax(0, Kind::Float);
    for (corr, loss) in corrs.iter_mut().zip(losses.iter::<f64>().unwrap()) {
        corr.lm_prob = loss;
        corr.prob = corr.kbd_prob * corr.lm_prob;
    }

    let probs_sum: f64 = corrs.iter().map(|c| c.prob).sum();
    for corr in &mut corrs {
        corr.prob /= probs_sum;
    }

    // note how b is first, not a: this sorts in descending order
    corrs.sort_unstable_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());
    corrs
}

#[cfg(test)]
mod tests {
    use crate::{lm::LLM, model::KbdModel};

    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_symspell() {
        assert_eq!(
            check_suggestions("Pairs", 1)
                .into_iter()
                .map(|s| s.term)
                .collect::<Vec<_>>(),
            vec!["Pamirs", "Paris", "airs", "fairs", "hairs", "lairs", "pairs"]
        );
    }

    #[test]
    fn it_works() {
        // let orig = "otol";
        // let result = check_suggestions(orig, 2);
        //assert_eq!(result, vec![]);

        // let comps: Vec<String> = result
        //     .iter()
        //     .take(3)
        //     .map(|s| s.term.clone().into())
        //     .collect();
        // assert_eq!(comps, vec!["".to_string()]);

        // let ops: Vec<Vec<IndexedOperation<LevenshteinDamerauOp>>> = result
        //     .into_iter()
        //     .take(3)
        //     .map(|s| edit_ops(orig, s))
        //     .collect();

        // let test_ops: Vec<Vec<IndexedOperation<LevenshteinDamerauOp>>> = vec![];
        // assert_eq!(ops, test_ops);
    }

    #[test]
    fn test_corrections() {
        let orig = "otol";
        let lm = LLM.lock().unwrap();
        let corrs = corrections(orig, 1, &KbdModel::default(), None, &*lm);
        assert_eq!(
            corrs.into_iter().map(|c| c.term).collect::<Vec<String>>(),
            vec!["tool".to_string()]
        );

        let orig = "teh";
        let corrs = corrections(orig, 1, &KbdModel::default(), None, &*lm);
        let probs: Vec<f64> = corrs.iter().map(|c| c.prob).collect();
        let prob_sum: f64 = probs.iter().sum();
        let normed_probs: Vec<f64> = probs.iter().map(|x| x / prob_sum).collect();
        let terms: Vec<String> = corrs.into_iter().map(|c| c.term).collect();
        assert_eq!(
            terms[0],
            "the".to_string(),
            "\n{:?}\n{:?}",
            terms,
            normed_probs
        );
    }

    #[test]
    fn test_corrections_ctxt() {
        let orig = "Pairs";
        let lm = LLM.lock().unwrap();
        let corrs = corrections(
            orig,
            1,
            &KbdModel::default(),
            Some("The capital of France is "),
            &*lm,
        );
        let terms: Vec<String> = corrs.iter().map(|c| c.term.clone()).collect();
        assert_eq!(terms[0], "Pairs".to_string(), "\n{:#?}", corrs);
    }
}
