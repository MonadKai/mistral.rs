//! Sinusoidal positional encoding for the audio CNN stack.
//
// Derived from the Apache-2.0 `aha` reference (`position_embed/sinusoidal_pe.rs`).

use candle_core::{Device, Result, Tensor};

use super::rope_mrope::{compute_default_rope_parameters, sinusoidal_encode};

pub(crate) struct SinusoidalPositionEncoderCat {
    inv_freq: Option<Tensor>,
}

impl SinusoidalPositionEncoderCat {
    pub fn new(dim: Option<usize>, save_freq: bool, device: &Device) -> Result<Self> {
        let inv_freq = if save_freq {
            if let Some(dim) = dim {
                let v = compute_default_rope_parameters(dim, 10000.0);
                Some(Tensor::from_slice(&v, (1, v.len()), device)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self { inv_freq })
    }

    pub fn forward(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_, seq_len, head_dim) = xs.dims3()?;
        let pos_embed = sinusoidal_encode(
            seqlen_offset,
            seq_len,
            head_dim,
            xs.device(),
            xs.dtype(),
            self.inv_freq.as_ref(),
        )?
        .unsqueeze(0)?;
        xs.broadcast_add(&pos_embed)
    }
}
