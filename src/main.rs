use aispell::checker::EnChecker;
use aispell::{corrections, lm::LLM, model::KbdModel};
use anyhow;
use iced::executor;
use iced::{Application, Command, Element, Settings, Theme};

fn main() -> iced::Result {
    // let orig = "Pairs";
    // let lm = LLM.lock().unwrap();
    // let corrs = corrections(
    //     orig,
    //     1,
    //     &KbdModel::default(),
    //     Some("The capital of France is "),
    //     &*lm,
    // );
    // println!("{:#?}", corrs);
    EnChecker::run(Settings::default())
}
