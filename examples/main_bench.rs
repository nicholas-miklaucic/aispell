use aispell::{corrections, model::KbdModel, rwkv::RwkvLM};

pub fn main() {
    println!("Testing");
    let lm = RwkvLM::try_new().unwrap();
    let km = KbdModel::default();

    let l = corrections("three", 2, &km, Some("One plus two makes "), &lm).len();
    println!("{}", l);
}
