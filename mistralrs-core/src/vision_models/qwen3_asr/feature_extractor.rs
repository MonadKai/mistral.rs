//! Whisper-compatible log-mel frontend (`preprocessor_config.json`).
//
// STFT/mel path mirrors `vision_models::voxtral::audio_processing` (Slaney);
// normalization follows the Apache-2.0 `aha` Whisper extractor.

use candle_core::Result;
use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};

use super::tensor_utils::pad_reflect_last_dim;

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct WhisperPreprocessorConfig {
    pub chunk_length: Option<f32>,
    pub dither: f64,
    pub feature_extractor_type: Option<String>,
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
}

fn default_sampling_rate() -> usize {
    16000
}

impl WhisperPreprocessorConfig {
    /// Deserialize the merged HF [`crate::vision_models::preprocessor_config::PreProcessorConfig`]
    /// (from `preprocessor_config.json`) into this checkpoint-specific shape.
    pub(crate) fn try_from_preprocessor(
        p: &crate::vision_models::preprocessor_config::PreProcessorConfig,
    ) -> anyhow::Result<Self> {
        let n_fft = p.n_fft.or(p.fft_length).ok_or_else(|| {
            anyhow::anyhow!("preprocessor_config: missing `n_fft` / `fft_length`")
        })?;
        Ok(Self {
            chunk_length: p.chunk_length,
            dither: p.dither.unwrap_or(0.0),
            feature_extractor_type: p.feature_extractor_type.clone(),
            feature_size: p
                .feature_size
                .ok_or_else(|| anyhow::anyhow!("preprocessor_config: missing `feature_size`"))?,
            hop_length: p
                .hop_length
                .ok_or_else(|| anyhow::anyhow!("preprocessor_config: missing `hop_length`"))?,
            n_fft,
            sampling_rate: p.sampling_rate.unwrap_or(16000),
        })
    }
}

pub(crate) struct WhisperFeatureExtractor {
    hop_length: usize,
    n_fft: usize,
    dither: f64,
    sampling_rate: usize,
    mel_filters: Vec<Vec<f32>>,
}

impl WhisperFeatureExtractor {
    pub fn new(cfg: &WhisperPreprocessorConfig, _device: &Device) -> Result<Self> {
        let mel_filters = create_mel_filterbank(cfg.n_fft, cfg.feature_size, cfg.sampling_rate)?;
        Ok(Self {
            hop_length: cfg.hop_length,
            n_fft: cfg.n_fft,
            dither: cfg.dither,
            sampling_rate: cfg.sampling_rate,
            mel_filters,
        })
    }

    pub fn sampling_rate(&self) -> usize {
        self.sampling_rate
    }

    /// `[1, n_mels, n_frames]` matching HF `WhisperFeatureExtractor` output before squeeze.
    pub fn compute(&self, waveform_batched: &Tensor) -> Result<Tensor> {
        let mut waveform = waveform_batched.clone();
        if self.dither != 0.0 {
            waveform = waveform.add(&waveform.randn_like(0.0, 1.0)?.affine(self.dither, 0.0)?)?;
        }
        let pad = self.n_fft / 2;
        let waveform = pad_reflect_last_dim(&waveform, (pad, pad))?;
        let n_fft = self.n_fft;
        let hop = self.hop_length;
        let n_freqs = n_fft / 2 + 1;

        let window: Vec<f32> = (0..n_fft)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos()))
            .collect();

        let total_frames = (waveform.dim(1)? - n_fft) / hop + 1;
        let num_frames = total_frames.saturating_sub(1).max(1);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        let wf = waveform.flatten_all()?.to_vec1::<f32>()?;
        let padded_len = wf.len();

        let mut mel_features = Vec::with_capacity(num_frames);
        let log_mel_floor = -8.0_f32;

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop;
            if start + n_fft > padded_len {
                break;
            }
            let mut buf: Vec<Complex32> = wf[start..start + n_fft]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex32::new(s * w, 0.0))
                .collect();

            fft.process(&mut buf);

            let power: Vec<f32> = buf[..n_freqs].iter().map(|c| c.norm_sqr()).collect();

            let mut mel_frame = vec![0.0f32; self.mel_filters.len()];
            for (mel_idx, filter) in self.mel_filters.iter().enumerate() {
                let mut sum = 0.0f32;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < power.len() {
                        sum += power[freq_idx] * coeff;
                    }
                }
                let log_val = sum.max(1e-10).log10();
                let clamped = log_val.max(log_mel_floor);
                mel_frame[mel_idx] = (clamped + 4.0) / 4.0;
            }

            mel_features.push(mel_frame);
        }

        if mel_features.is_empty() {
            candle_core::bail!("audio too short for mel features");
        }

        let num_frames = mel_features.len();
        let n_mels = self.mel_filters.len();
        let data: Vec<f32> = mel_features.into_iter().flatten().collect();
        let on_dev = waveform.device();
        let tensor = Tensor::from_vec(data, (1, num_frames, n_mels), on_dev)?;
        tensor.permute((0, 2, 1))
    }
}

fn hertz_to_mel(freq: f32) -> f32 {
    const MIN_LOG_HERTZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOGSTEP: f32 = 27.0 / 1.856_298;
    if freq >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HERTZ).ln() * LOGSTEP
    } else {
        3.0 * freq / 200.0
    }
}

fn mel_to_hertz(mel: f32) -> f32 {
    const MIN_LOG_HERTZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    const LOGSTEP: f32 = 1.856_298 / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

fn create_mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sampling_rate: usize,
) -> Result<Vec<Vec<f32>>> {
    let n_freqs = n_fft / 2 + 1;
    let sr = sampling_rate as f32;

    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * (sr / 2.0) / (n_freqs - 1).max(1) as f32)
        .collect();

    let mel_min = hertz_to_mel(0.0);
    let mel_max = hertz_to_mel(sr / 2.0);
    let filter_freqs: Vec<f32> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_to_hertz(mel)
        })
        .collect();

    let filter_diff: Vec<f32> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];
    for m in 0..n_mels {
        for (j, &fft_f) in fft_freqs.iter().enumerate() {
            let slope_left = fft_f - filter_freqs[m];
            let slope_right = filter_freqs[m + 2] - fft_f;
            let down = slope_left / filter_diff[m];
            let up = slope_right / filter_diff[m + 1];
            filterbank[m][j] = 0.0f32.max(down.min(up));
        }
    }

    for m in 0..n_mels {
        let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
        for val in filterbank[m].iter_mut().take(n_freqs) {
            *val *= enorm;
        }
    }

    Ok(filterbank)
}
