//! Attention/MLP/Conv helpers ported from the Apache-2.0 `aha` reference (`models/common/modules.rs`).

use candle_core::Result;
use candle_core::{Tensor, D};
use candle_nn::{
    ops::softmax_last_dim, Activation, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear,
    Module,
};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::RmsNorm;

use super::rope_mrope::apply_rotary_pos_emb;
use super::tensor_utils::repeat_kv;
use super::weights::{self};

pub(crate) struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GateUpDownMLP {
    pub fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: weights::linear(vb.pp("gate_proj"), hidden_size, intermediate_size, false)?,
            up_proj: weights::linear(vb.pp("up_proj"), hidden_size, intermediate_size, false)?,
            down_proj: weights::linear(vb.pp("down_proj"), intermediate_size, hidden_size, false)?,
            act_fn,
        })
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub(crate) struct QKNormAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QKNormAttention {
    pub fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        head_dim: Option<usize>,
        num_key_value_heads: Option<usize>,
        attention_bias: bool,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let head_dim = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        Ok(Self {
            q_proj: weights::linear(
                vb.pp("q_proj"),
                hidden_size,
                num_attention_heads * head_dim,
                attention_bias,
            )?,
            k_proj: weights::linear(
                vb.pp("k_proj"),
                hidden_size,
                num_key_value_heads * head_dim,
                attention_bias,
            )?,
            v_proj: weights::linear(
                vb.pp("v_proj"),
                hidden_size,
                num_key_value_heads * head_dim,
                attention_bias,
            )?,
            o_proj: weights::linear(
                vb.pp("o_proj"),
                num_attention_heads * head_dim,
                hidden_size,
                attention_bias,
            )?,
            q_norm: RmsNorm::new(head_dim, rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, rms_norm_eps, vb.pp("k_norm"))?,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[prev_k, &key_states], 2)?,
                Tensor::cat(&[prev_v, &value_states], 2)?,
            ),
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;
        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        attn_output.apply(&self.o_proj)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

pub(crate) struct NaiveAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
}

impl NaiveAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: Option<usize>,
        bias: bool,
        q_proj_pp_name: Option<&str>,
        k_proj_pp_name: Option<&str>,
        v_proj_pp_name: Option<&str>,
        o_proj_pp_name: Option<&str>,
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let q_proj_pp_name = q_proj_pp_name.unwrap_or("q_proj");
        let k_proj_pp_name = k_proj_pp_name.unwrap_or("k_proj");
        let v_proj_pp_name = v_proj_pp_name.unwrap_or("v_proj");
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        Ok(Self {
            q_proj: weights::linear(
                vb.pp(q_proj_pp_name),
                hidden_size,
                num_attention_heads * head_dim,
                bias,
            )?,
            k_proj: weights::linear(
                vb.pp(k_proj_pp_name),
                hidden_size,
                num_key_value_heads * head_dim,
                bias,
            )?,
            v_proj: weights::linear(
                vb.pp(v_proj_pp_name),
                hidden_size,
                num_key_value_heads * head_dim,
                bias,
            )?,
            o_proj: weights::linear(
                vb.pp(o_proj_pp_name),
                num_attention_heads * head_dim,
                hidden_size,
                bias,
            )?,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: num_attention_heads * head_dim,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        to_f32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = if let (Some(cos), Some(sin)) = (cos, sin) {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, to_f32)?
        } else {
            (query_states, key_states)
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        attn_output.apply(&self.o_proj)
    }
}

pub(crate) fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    let key_states = match num_key_value_groups {
        Some(g) => repeat_kv(key_states.clone(), g)?.contiguous()?,
        None => key_states.clone(),
    };
    let value_states = match num_key_value_groups {
        Some(g) => repeat_kv(value_states.clone(), g)?.contiguous()?,
        None => value_states.clone(),
    };
    let query_states = query_states.contiguous()?;
    let mut attn_weights = query_states.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
    attn_weights = (attn_weights * scaling)?;
    if let Some(mask) = attention_mask {
        attn_weights = attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?;
    }
    let attn_weights = softmax_last_dim(&attn_weights)?;
    let attn_output = attn_weights.matmul(&value_states)?;
    attn_output.transpose(1, 2)?.contiguous()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn get_conv2d(
    vb: ShardedVarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding,
        stride,
        dilation,
        groups,
        cudnn_fwd_algo: None,
    };
    weights::conv2d(vb, cfg, in_c, out_c, kernel_size, bias)
}

pub(crate) fn get_layer_norm(
    vb: ShardedVarBuilder,
    eps: f64,
    dim: usize,
    affine: bool,
) -> Result<LayerNorm> {
    let ln_config = LayerNormConfig {
        eps,
        remove_mean: true,
        affine,
    };
    crate::layers::layer_norm(dim, ln_config, vb)
}
