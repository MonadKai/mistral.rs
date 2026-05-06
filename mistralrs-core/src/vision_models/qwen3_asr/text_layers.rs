//! Qwen3 decoder blocks shared by Qwen3-ASR text tower.
//
// Derived from the Apache-2.0 `aha` reference (`models/qwen3/model.rs`).

use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Embedding, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::{embedding, RmsNorm};

use super::config::Qwen3TextCfg;
use super::nn::{GateUpDownMLP, QKNormAttention};
use super::tensor_utils::prepare_causal_attention_mask;
use super::weights;

pub(crate) struct Qwen3DecoderLayer {
    self_attn: QKNormAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    pub fn new(config: &Qwen3TextCfg, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: QKNormAttention::new(
                vb.pp("self_attn"),
                config.hidden_size,
                config.num_attention_heads,
                Some(config.head_dim),
                Some(config.num_key_value_heads),
                config.attention_bias,
                config.rms_norm_eps,
            )?,
            mlp: GateUpDownMLP::new(
                vb.pp("mlp"),
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
            )?,
            input_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual.add(&xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub(crate) struct Qwen3ASRThinkerTextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen3DecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: super::rope_mrope::Qwen3VLTextRotaryEmbedding,
    pub mrope_section: Vec<usize>,
}

impl Qwen3ASRThinkerTextModel {
    pub fn new(vb: ShardedVarBuilder, cfg: &super::config::Qwen3ASRTextConfig) -> Result<Self> {
        let qcfg = super::config::qwen3asr_text_config_to_qwen3_text_cfg(cfg);
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embed_tokens"),
            &None,
        )?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(&qcfg, vb_layers.pp(i))?;
            layers.push(layer);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb =
            super::rope_mrope::Qwen3VLTextRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section: cfg.rope_scaling.mrope_section.clone(),
        })
    }

    pub fn forward(
        &mut self,
        input_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = input_embeds.dims3()?;
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                input_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        let (cos, sin) = self.rotary_emb.forward_asr(
            &position_ids,
            input_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        let mut xs = input_embeds.clone();
        let attention_mask: Option<Tensor> = if seq_len <= 1 {
            None
        } else {
            Some(prepare_causal_attention_mask(
                b_size,
                seq_len,
                0,
                input_embeds.device(),
            )?)
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        }
        self.norm.forward(&xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

pub(crate) fn maybe_lm_head(
    vb: ShardedVarBuilder,
    cfg: &super::config::Qwen3ASRTextConfig,
    embed: &Embedding,
) -> Result<Linear> {
    if cfg.tie_word_embeddings {
        Ok(Linear::new(embed.embeddings().clone(), None))
    } else {
        weights::linear(vb, cfg.hidden_size, cfg.vocab_size, false)
    }
}
