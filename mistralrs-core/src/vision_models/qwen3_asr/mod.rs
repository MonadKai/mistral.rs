//! Qwen3-ASR (`Qwen3ASRForConditionalGeneration`) — speech encoder + Qwen3 text decoder.
//!
//! Ported from the Apache-2.0 [`aha`](https://github.com/) reference implementation
//! (see repository `NOTICE` in this worktree / upstream mistral.rs as applicable).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod config;
mod feature_extractor;
mod inner;
mod inputs_processor;
mod model;
mod nn;
mod rope_mrope;
mod sinusoidal;
mod tensor_utils;
mod text_layers;
mod weights;

// Public surface for loaders / processor wiring (not always referenced inside mistralrs-core).
#[allow(unused_imports)]
pub use config::{Qwen3ASRConfig, Qwen3ASRGenerationConfig, ThinkerConfig};
#[allow(unused_imports)]
pub use feature_extractor::WhisperPreprocessorConfig;
#[allow(unused_imports)]
pub use inputs_processor::Qwen3AsrProcessor;
#[allow(unused_imports)]
pub use model::{Qwen3AsrModel, Qwen3AsrSpecificArgs};
