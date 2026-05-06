//! Inputs processor + [`crate::pipeline::Processor`] wiring for Qwen3-ASR.
//!
//! Mel spectrograms are stored in `pixel_values` with shape `[1, n_mels, time]`.

use std::sync::Arc;

use anyhow::Result;
use candle_core::{Device, Tensor};
use mistralrs_audio::AudioInput;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use tokenizers::Tokenizer;

use crate::device_map::DeviceMapper;
use crate::pipeline::{
    text_models_inputs_processor::{get_completion_input, get_prompt_input, PagedAttentionMeta},
    InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
};
use crate::sequence::Sequence;
use crate::vision_models::ModelInputs;

use super::config::Qwen3ASRConfig;
use super::feature_extractor::{WhisperFeatureExtractor, WhisperPreprocessorConfig};
use super::model::Qwen3AsrSpecificArgs;
use super::tensor_utils::get_feat_extract_output_lengths;

#[derive(Clone)]
pub struct Qwen3AsrProcessor {
    inner: Arc<Qwen3AsrInputsInner>,
}

struct Qwen3AsrInputsInner {
    extractor: WhisperFeatureExtractor,
    audio_token_id: u32,
}

impl Qwen3AsrProcessor {
    pub fn new(
        cfg: &Qwen3ASRConfig,
        preprocessor: &WhisperPreprocessorConfig,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Qwen3AsrInputsInner {
                extractor: WhisperFeatureExtractor::new(preprocessor, device)?,
                audio_token_id: cfg.thinker_config.audio_token_id,
            }),
        })
    }

    fn audio_to_waveform_tensor(
        audio: &AudioInput,
        target_sr: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mono = audio.to_mono();
        let samples = if audio.sample_rate != target_sr as u32 {
            Self::resample(&mono, audio.sample_rate, target_sr as u32)?
        } else {
            mono
        };
        let n = samples.len();
        Tensor::from_vec(samples, (1, n), device).map_err(anyhow::Error::from)
    }

    fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(samples.to_vec());
        }
        let sinc = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(
            to_rate as f64 / from_rate as f64,
            2.0,
            sinc,
            samples.len(),
            1,
        )?;
        let result = resampler.process(&[samples.to_vec()], None)?;
        Ok(result[0].clone())
    }

    fn expand_audio_token_run(ids: Vec<u32>, audio_tok: u32, n: usize) -> Vec<u32> {
        let mut out = Vec::new();
        let mut done = false;
        for t in ids {
            if t == audio_tok && !done {
                for _ in 0..n {
                    out.push(audio_tok);
                }
                done = true;
            } else {
                out.push(t);
            }
        }
        out
    }
}

impl Processor for Qwen3AsrProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Qwen3AsrInputsProcessor {
            inner: self.inner.clone(),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<|audio_pad|>", "<|audio_start|>", "<|audio_end|>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

struct Qwen3AsrInputsProcessor {
    inner: Arc<Qwen3AsrInputsInner>,
}

impl InputsProcessor for Qwen3AsrInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        _other_config: Option<Arc<dyn std::any::Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let _ = tokenizer;
        if is_xlora {
            anyhow::bail!("X-LoRA unsupported for Qwen3-ASR.");
        }
        if no_kv_cache {
            anyhow::bail!("Qwen3-ASR requires KV cache.");
        }

        let mel_tensor = if is_prompt {
            let mut mel_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                if seq.has_audios() {
                    if let Some(audios) = seq.take_audios() {
                        for audio in audios {
                            let wf = Qwen3AsrProcessor::audio_to_waveform_tensor(
                                &audio,
                                self.inner.extractor.sampling_rate(),
                                device,
                            )?;
                            let mel = self.inner.extractor.compute(&wf)?;
                            mel_accum.push(mel);
                        }
                    }
                }
            }
            match mel_accum.len() {
                0 => None,
                1 => Some(mel_accum[0].clone()),
                _ => Some(Tensor::cat(&mel_accum.iter().collect::<Vec<_>>(), 0)?),
            }
        } else {
            None
        };

        let inner = if is_prompt {
            get_prompt_input(
                input_seqs.iter().map(|s| s.get_toks()).collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )?
        } else {
            get_completion_input(
                input_seqs.iter().map(|s| s.get_toks()).collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )?
        };

        let mut input_ids = inner.inputs.input;
        let pixel_values = if let Some(ref mel) = mel_tensor {
            if input_ids.dim(0)? != 1 {
                anyhow::bail!("Qwen3-ASR currently supports batch size 1.");
            }
            let n_frames = mel.dim(2)?;
            let out_len = get_feat_extract_output_lengths(n_frames);
            let row = input_ids.to_vec2::<u32>()?.pop().unwrap_or_default();
            let expanded =
                Qwen3AsrProcessor::expand_audio_token_run(row, self.inner.audio_token_id, out_len);
            let len = expanded.len();
            input_ids = Tensor::from_vec(expanded, (1, len), device)?;
            Some(mel.clone())
        } else {
            None
        };

        let inputs: Box<dyn std::any::Any> = Box::new(ModelInputs {
            input_ids,
            seqlen_offsets: inner.inputs.positions,
            context_lens: inner.inputs.context_lens,
            position_ids: inner.inputs.position_ids,
            pixel_values,
            model_specific_args: Box::new(Qwen3AsrSpecificArgs {
                input_features: mel_tensor.clone(),
            }),
            paged_attn_meta: inner.inputs.paged_attn_meta,
            flash_meta: inner.inputs.flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices: inner.seq_indices,
        })
    }
}
