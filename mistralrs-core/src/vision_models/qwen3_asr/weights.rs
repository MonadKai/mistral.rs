//! Load `candle_nn` modules directly from [`mistralrs_quant::ShardedVarBuilder`] tensors.

use candle_core::Result;
use candle_nn::{Conv2d, Conv2dConfig, Linear};
use mistralrs_quant::ShardedVarBuilder;

pub(crate) fn linear(
    vb: ShardedVarBuilder,
    in_features: usize,
    out_features: usize,
    bias: bool,
) -> Result<Linear> {
    let weight = vb.get((out_features, in_features), "weight")?;
    let bias_tensor = if bias {
        Some(vb.get(out_features, "bias")?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias_tensor))
}

pub(crate) fn conv2d(
    vb: ShardedVarBuilder,
    mut cfg: Conv2dConfig,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    bias: bool,
) -> Result<Conv2d> {
    let w = vb.get(
        (out_channels, in_channels, kernel_size, kernel_size),
        "weight",
    )?;
    let b = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    cfg.cudnn_fwd_algo = None;
    Ok(Conv2d::new(w, b, cfg))
}
