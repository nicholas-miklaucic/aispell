//! Spell checker with debug

use crate::{
    corrections,
    gpt_lm::{GptLM, LLM},
    lm::LM,
    model::{KbdModel, Model},
    Correction,
};
use iced::widget::{text_input, Column};
use iced::Element;
use iced::{
    widget::{column, container},
    Length, Sandbox,
};
use iced::{
    widget::{row, text},
    Theme,
};
use lazy_static::lazy_static;

#[derive(Clone, Debug)]
/// The state of the checker.
pub struct Checker<K: Model<u8>, L: LM<u8>> {
    /// The current text.
    pub text: String,

    /// The keyboard model.
    km: K,

    /// The language model.
    lm: L,
}

pub type EnChecker = Checker<KbdModel, GptLM>;

impl EnChecker {
    pub fn new() -> Self {
        let lm = GptLM::try_new().unwrap();
        let km = KbdModel::default();
        let text = "Truly intelligent spell chucking".to_string();

        Self { text, km, lm }
    }

    pub fn compute_corrections(&self) -> Vec<Correction> {
        let words: Vec<&str> = self.text.trim_end_matches(' ').split(' ').collect();

        let (prev, term) = match words.as_slice() {
            [] => return vec![],
            [t] => ("".to_string(), format!(" {}", t)),
            [p @ .., t] => (p.join(" ").to_string(), format!(" {}", t)),
        };

        corrections(&term, 2, &self.km, Some(&prev), &self.lm)
    }
}

#[derive(Debug, Clone)]
/// The messages.
pub enum Message {
    InputChanged(String),
}

lazy_static! {
    pub static ref INPUT_ID: text_input::Id = text_input::Id::unique();
}

impl Sandbox for EnChecker {
    type Message = Message;

    fn new() -> Self {
        Self::new()
    }

    fn title(&self) -> String {
        "AISpell".into()
    }

    fn update(&mut self, message: Self::Message) {
        match message {
            Message::InputChanged(s) => {
                self.text = s;
            }
        }
    }

    fn view(&self) -> Element<Self::Message> {
        let input = text_input("Type text to correct", &self.text, Message::InputChanged)
            .id(INPUT_ID.clone())
            .padding(15);

        let results: Vec<_> = self
            .compute_corrections()
            .iter()
            .take(10)
            .map(
                |Correction {
                     term,
                     prob,
                     lm_prob,
                     kbd_prob,
                 }| {
                    row![
                        text(term),
                        text(format!("{:.3}", prob)),
                        text(format!("{:.3}", lm_prob)),
                        text(format!("{:.3}", kbd_prob))
                    ]
                    .spacing(30)
                    .into()
                },
            )
            .collect();

        let body = Column::with_children(results).spacing(20).max_width(1200);

        let content = column![input, body].spacing(20).max_width(1200);
        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(40)
            .center_x()
            .into()
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }
}
