//! MRoPE helpers for Qwen3-ASR text decoder (`forward_asr`).
//
// Derived from the Apache-2.0 `aha` reference (`position_embed/rope.rs`).

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

pub(crate) fn compute_default_rope_parameters(dim: usize, base: f32) -> Vec<f32> {
    (0..dim)
        .step_by(2)
        .map(|i| 1.0_f32 / base.powf(i as f32 / dim as f32))
        .collect()
}

pub(crate) fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    let x2 = x2.affine(-1.0, 0.0)?;
    Tensor::cat(&[&x2, &x1], D::Minus1)?.contiguous()
}

pub(crate) fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    to_f32: bool,
) -> Result<(Tensor, Tensor)> {
    let mut cos = cos.clone();
    let mut sin = sin.clone();
    if cos.rank() == 2 {
        cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos.rank() == 3 {
        cos = cos.unsqueeze(1)?;
        sin = sin.unsqueeze(1)?;
    }
    let orig_dtype = q.dtype();
    let q_work = if to_f32 {
        q.to_dtype(DType::F32)?
    } else {
        q.clone()
    };
    let k_work = if to_f32 {
        k.to_dtype(DType::F32)?
    } else {
        k.clone()
    };
    let cos = cos.to_dtype(q_work.dtype())?;
    let sin = sin.to_dtype(q_work.dtype())?;

    let q_embed = q_work
        .broadcast_mul(&cos)?
        .add(&rotate_half(&q_work)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    let k_embed = k_work
        .broadcast_mul(&cos)?
        .add(&rotate_half(&k_work)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    Ok((q_embed, k_embed))
}

/// Rotary embedding used by Qwen3-VL family; ASR uses [`Self::forward_asr`].
#[derive(Debug, Clone)]
pub struct Qwen3VLTextRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen3VLTextRotaryEmbedding {
    pub fn new(dim: usize, theta_base: f32) -> Self {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }

    fn apply_interleaved_mrope_asr(
        &self,
        freqs: &Tensor,
        mrope_section: Vec<usize>,
    ) -> Result<Tensor> {
        let mut freqs_t = freqs.i(0)?.contiguous()?;
        for (dim, _offset) in (1..3).enumerate() {
            let dim = dim + 1;
            let length = mrope_section[dim];
            let idx = Tensor::arange_step(dim as u32, length as u32, 3, freqs.device())?;
            let src = freqs.i(dim)?.contiguous()?;
            let src = src.index_select(&idx, D::Minus1)?.contiguous()?;
            let idx = idx
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as(src.shape())?
                .contiguous()?;
            freqs_t = freqs_t.scatter(&idx, &src, D::Minus1)?;
        }
        Ok(freqs_t)
    }

    pub fn forward_asr(
        &self,
        position_ids: &Tensor,
        dtype: DType,
        mrope_section: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let position_ids = if position_ids.rank() == 2 {
            let (bs, len) = position_ids.dims2()?;
            position_ids.unsqueeze(0)?.expand((3, bs, len))?
        } else {
            position_ids.clone()
        };
        let position_ids_expanded = position_ids
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let inv_freq_expanded = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, self.inv_freq.len(), 1),
            position_ids.device(),
        )?
        .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
        .to_dtype(DType::F32)?
        .contiguous()?;

        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded)?
            .transpose(2, 3)?;
        let freqs = self.apply_interleaved_mrope_asr(&freqs, mrope_section)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
    }
}

pub(crate) fn sinusoidal_encode(
    seqlen_offset: usize,
    seq_len: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
    inv_freq_buf: Option<&Tensor>,
) -> Result<Tensor> {
    let positions = Tensor::arange(
        seqlen_offset as f32,
        (seqlen_offset + seq_len) as f32,
        device,
    )?
    .reshape((seq_len, 1))?;
    let inv_freq = if let Some(inv_freq) = inv_freq_buf {
        inv_freq.clone()
    } else {
        let inv_freq = compute_default_rope_parameters(head_dim, 10000.0);
        Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?
    };
    let freqs = positions.matmul(&inv_freq)?;
    let sin = freqs.sin()?;
    let cos = freqs.cos()?;
    Tensor::cat(&[sin, cos], D::Minus1)?
        .contiguous()?
        .to_dtype(dtype)
}
