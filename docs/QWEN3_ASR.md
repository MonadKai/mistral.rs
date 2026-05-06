# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) is loaded as a multimodal pipeline with architecture name **`qwen3_asr`** or **`Qwen3ASRForConditionalGeneration`**.

## Inputs

- **Audio**: attach [`AudioInput`](https://docs.rs/mistralrs-audio) on the user message (same pattern as [Voxtral](VOXTRAL.md)).
- **Mel / encoder conditioning**: the pipeline passes log-mel features in **`pixel_values`** with shape `[1, n_mels, time]`, matching the field name used for vision models. The same tensor can be supplied as `Qwen3AsrSpecificArgs::input_features` in `model_specific_args` if you extend the stack.
- **Placeholders**: the chat template may include a single `audio_token_id` run; the inputs processor expands that run to the encoder output length (Whisper-style conv stack) so token count matches `audio_tower` frame count.

## Preprocessor

`preprocessor_config.json` must match the official checkpoint (e.g. `WhisperFeatureExtractor` with `feature_size`, `hop_length`, `n_fft`). These keys are merged into `mistralrs-core` `PreProcessorConfig` and mapped to the internal `WhisperPreprocessorConfig`.

## Running locally

```bash
export QWEN3_ASR_MODEL=/path/to/Qwen3-ASR-0.6B
export QWEN3_ASR_SAMPLE=/path/to/sample.wav
cargo run --release -p mistralrs --example qwen3_asr
```

Defaults in the example point at a common ModelScope cache layout and `sample_speech.wav` in the repo root when env vars are unset.
