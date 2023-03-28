use std::{
    sync::{mpsc, Mutex},
    thread::{self, JoinHandle},
};

use aispell::{
    corrections,
    lm::{GptLM, LLM, LM},
    model::{KbdModel, Model},
    onnx_lm::OnnxLM,
    Correction,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::{sync::oneshot, task};
use warp::{hyper::Method, Filter};

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct CorrectionsRequest {
    text: String,
    edit_dist: i64,
}

// Message type for internal channel, passing around texts and return value
/// senders
type Message = (CorrectionsRequest, oneshot::Sender<Vec<Correction>>);

/// Runner for sentiment classification
#[derive(Debug, Clone)]
pub struct Corrector {
    sender: mpsc::SyncSender<Message>,
}

impl Corrector {
    /// Spawn a classifier on a separate thread and return a classifier instance
    /// to interact with it
    pub fn spawn() -> (JoinHandle<Result<()>>, Self) {
        let (sender, receiver) = mpsc::sync_channel(100);
        let handle = thread::spawn(move || Self::runner(receiver));
        (handle, Self { sender })
    }

    /// The classification runner itself
    fn runner(receiver: mpsc::Receiver<Message>) -> Result<()> {
        // Needs to be in sync runtime, async doesn't work
        let lm = OnnxLM::try_new_cached("bloom-560m").unwrap();
        let km = KbdModel::default();

        while let Ok((req, sender)) = receiver.recv() {
            let words: Vec<&str> = req.text.trim_end_matches(' ').split(' ').collect();

            let (prev, term) = match words.as_slice() {
                [] => (". ".to_string(), "".to_string()),
                [t] => (". ".to_string(), format!("{}", t)),
                [p @ .., t] => (format!("{} ", p.join(" ")), format!("{}", t)),
            };

            let reply = corrections(&term, req.edit_dist, &km, Some(&prev), &lm);
            sender.send(reply).expect("sending results");
        }

        Ok(())
    }
}
/// Make the runner predict a sample and return the result
pub async fn correct(
    model: Corrector,
    req: CorrectionsRequest,
) -> Result<impl warp::Reply, warp::reject::Rejection> {
    let (sender, receiver) = oneshot::channel();
    match task::block_in_place(|| model.sender.send((req, sender))) {
        Ok(()) => match receiver.await {
            Ok(corr) => Ok(warp::reply::with_header(
                warp::reply::with_header(
                    warp::reply::json(&corr),
                    "Access-Control-Allow-Origin",
                    "*",
                ),
                "Access-Control-Allow-Credentials",
                "true",
            )),
            Err(_e) => Err(warp::reject()),
        },
        Err(_e) => Err(warp::reject()),
    }
}

#[tokio::main]
async fn main() {
    let (_handle, corrector) = Corrector::spawn();
    let correct = warp::post()
        .and(warp::path("corrections"))
        // Only accept bodies smaller than 16kb...
        .and(warp::body::content_length_limit(1024 * 16))
        .map(move || corrector.clone())
        .and(warp::body::json())
        .and_then(correct)
        .with(
            warp::cors()
                .allow_methods(&[Method::POST])
                .allow_headers(vec![
                    "Access-Control-Allow-Origin",
                    "Origin",
                    "Accept",
                    "X-Requested-With",
                    "Content-Type",
                ])
                .allow_any_origin(),
        );

    warp::serve(correct).run(([127, 0, 0, 1], 3030)).await;
}
