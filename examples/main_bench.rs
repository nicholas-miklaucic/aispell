use aispell::{corrections, model::KbdModel, onnx_lm::OnnxLM};

pub fn main() {
    println!("Testing");
    let lm = OnnxLM::try_new_cached("bloom-560m").unwrap();
    let km = KbdModel::default();

    let l = corrections("three", 2, &km, Some("One plus two makes "), &lm).len();
    println!("{}", l);
}
