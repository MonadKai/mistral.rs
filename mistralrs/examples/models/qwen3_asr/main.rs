//! Transcription with Qwen3-ASR (`Qwen3ASRForConditionalGeneration`).
//!
//! ```bash
//! export QWEN3_ASR_MODEL=/path/to/Qwen3-ASR-0.6B
//! export QWEN3_ASR_SAMPLE=sample.wav
//! cargo run --release -p mistralrs --example qwen3_asr
//! ```

use anyhow::Result;
use mistralrs::{AudioInput, MultimodalMessages, MultimodalModelBuilder, TextMessageRole};

#[tokio::main]
async fn main() -> Result<()> {
    let model_path = std::env::var("QWEN3_ASR_MODEL").unwrap_or_else(|_| {
        "/Users/admin/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B".to_string()
    });
    let wav_path =
        std::env::var("QWEN3_ASR_SAMPLE").unwrap_or_else(|_| "sample_speech.wav".to_string());

    let model = MultimodalModelBuilder::new(model_path)
        .with_logging()
        .build()
        .await?;

    let audio_bytes = std::fs::read(&wav_path)?;
    let audio = AudioInput::from_bytes(&audio_bytes)?;

    let messages = MultimodalMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Transcribe this audio.",
        vec![],
        vec![audio],
        vec![],
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
