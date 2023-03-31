//! Defines the core `LM` trait and an implementation thereof.

/// A language model that gives probabilities for words in context, with letters of type `T`.
pub trait LM<T> {
    /// Gets the loss of the given completion following the given context.
    fn loss(&self, context: &[T], completion: &[T]) -> f64;
}
