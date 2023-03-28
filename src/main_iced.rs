use aispell::checker::EnChecker;
use iced::{Application, Settings};

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
