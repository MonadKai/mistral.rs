//! Audio encoder + Qwen3 text thinker (`thinker.*` weights).
//
// Derived from the Apache-2.0 `aha` reference (`models/qwen3_asr/model.rs`).

use candle_core::Result;
use candle_core::Tensor;
use candle_nn::{Activation, Conv2d, LayerNorm, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use super::config::{Qwen3ASRAudioConfig, ThinkerConfig};
use super::nn::{get_conv2d, get_layer_norm, NaiveAttention};
use super::sinusoidal::SinusoidalPositionEncoderCat;
use super::tensor_utils::{
    get_equal_mask, get_feat_extract_output_lengths, masked_scatter_dim0, split_tensor,
    split_tensor_with_size,
};
use super::text_layers::{maybe_lm_head, Qwen3ASRThinkerTextModel};
use super::weights;

pub(crate) struct Qwen3ASRAudioEncoderLayer {
    self_attn: NaiveAttention,
    self_attn_layer_norm: LayerNorm,
    activation_fn: Activation,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl Qwen3ASRAudioEncoderLayer {
    pub fn new(vb: ShardedVarBuilder, config: &Qwen3ASRAudioConfig) -> Result<Self> {
        let self_attn = NaiveAttention::new(
            vb.pp("self_attn"),
            config.d_model,
            config.encoder_attention_heads,
            config.encoder_attention_heads,
            None,
            true,
            Some("q_proj"),
            Some("k_proj"),
            Some("v_proj"),
            Some("out_proj"),
        )?;
        let self_attn_layer_norm =
            get_layer_norm(vb.pp("self_attn_layer_norm"), 1e-5, config.d_model, true)?;
        let activation_fn = config.activation_function;
        let fc1 = weights::linear(vb.pp("fc1"), config.d_model, config.encoder_ffn_dim, true)?;
        let fc2 = weights::linear(vb.pp("fc2"), config.encoder_ffn_dim, config.d_model, true)?;
        let final_layer_norm =
            get_layer_norm(vb.pp("final_layer_norm"), 1e-5, config.d_model, true)?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            activation_fn,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None, None, mask, false)?;
        let residual = xs.add(&residual)?;
        let xs = self.final_layer_norm.forward(&residual)?;
        let xs = self.fc1.forward(&xs)?.apply(&self.activation_fn)?;
        let xs = self.fc2.forward(&xs)?;
        xs.add(&residual)
    }
}

pub(crate) struct Qwen3ASRAudioEncoder {
    n_window: usize,
    positional_embedding: SinusoidalPositionEncoderCat,
    layers: Vec<Qwen3ASRAudioEncoderLayer>,
    ln_post: LayerNorm,
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    proj1: Linear,
    act: Activation,
    proj2: Linear,
    conv_chunksize: usize,
}

impl Qwen3ASRAudioEncoder {
    #[allow(clippy::manual_div_ceil)]
    pub fn new(vb: ShardedVarBuilder, config: &Qwen3ASRAudioConfig) -> Result<Self> {
        let n_window = config.n_window;
        let positional_embedding =
            SinusoidalPositionEncoderCat::new(Some(config.d_model), true, vb.device())?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.encoder_layers {
            let layer = Qwen3ASRAudioEncoderLayer::new(vb_layers.pp(i), config)?;
            layers.push(layer);
        }
        let ln_post = get_layer_norm(vb.pp("ln_post"), 1e-5, config.d_model, true)?;
        let conv2d1 = get_conv2d(
            vb.pp("conv2d1"),
            1,
            config.downsample_hidden_size,
            3,
            1,
            2,
            1,
            1,
            true,
        )?;
        let conv2d2 = get_conv2d(
            vb.pp("conv2d2"),
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            1,
            2,
            1,
            1,
            true,
        )?;
        let conv2d3 = get_conv2d(
            vb.pp("conv2d3"),
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            1,
            2,
            1,
            1,
            true,
        )?;
        let in_dim =
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2);
        let conv_out = weights::linear(vb.pp("conv_out"), in_dim, config.d_model, false)?;
        let proj1 = weights::linear(vb.pp("proj1"), config.d_model, config.d_model, true)?;
        let act = config.activation_function;
        let proj2 = weights::linear(vb.pp("proj2"), config.d_model, config.output_dim, true)?;
        let conv_chunksize = config.conv_chunksize;
        Ok(Self {
            n_window,
            positional_embedding,
            layers,
            ln_post,
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            proj1,
            act,
            proj2,
            conv_chunksize,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let feature_lens = xs.dim(1)?;
        let chunk_num = feature_lens / (self.n_window * 2);
        let mut chunk_lengths = vec![self.n_window * 2; chunk_num];
        let last = feature_lens % (self.n_window * 2);
        if last > 0 {
            chunk_lengths.push(last);
        }
        let mut chunk_list = split_tensor(&xs.t()?, &chunk_lengths, 0)?;
        if last > 0 {
            let chunk_last = chunk_list
                .pop()
                .ok_or_else(|| candle_core::Error::Msg("chunk_list is empty".into()))?;
            let pad_size = self.n_window * 2 - last;
            let chunk_last = chunk_last.pad_with_zeros(0, 0, pad_size)?;
            chunk_list.push(chunk_last);
        }
        let padded_feature = Tensor::stack(&chunk_list, 0)?.transpose(1, 2)?;
        let feature_lens_after_cnn: Vec<usize> = chunk_lengths
            .iter()
            .map(|&i| get_feat_extract_output_lengths(i))
            .collect();
        let feature_len_after_cnn = feature_lens_after_cnn.iter().sum::<usize>();
        let padded_feature = padded_feature.unsqueeze(1)?;
        let mut padded_embeds = vec![];
        let feature_splits = split_tensor_with_size(&padded_feature, self.conv_chunksize, 0)?;
        for chunk in feature_splits.iter() {
            let padded_embed = self.conv2d1.forward(chunk)?.gelu()?;
            let padded_embed = self.conv2d2.forward(&padded_embed)?.gelu()?;
            let padded_embed = self.conv2d3.forward(&padded_embed)?.gelu()?;
            padded_embeds.push(padded_embed);
        }
        let padded_embed = Tensor::cat(&padded_embeds, 0)?;
        let (b, c, f, t) = padded_embed.dims4()?;
        let padded_embed =
            padded_embed
                .permute((0, 3, 1, 2))?
                .contiguous()?
                .reshape((b, t, c * f))?;
        let padded_embed = self.conv_out.forward(&padded_embed)?;
        let padded_embed = self.positional_embedding.forward(&padded_embed, 0)?;
        let padded_embed = padded_embed.flatten(0, 1)?;
        let mut hidden_states = padded_embed
            .narrow(0, 0, feature_len_after_cnn)?
            .unsqueeze(0)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, None)?;
        }
        let hidden_states = hidden_states.squeeze(0)?;
        let hidden_states = self.ln_post.forward(&hidden_states)?;
        let hidden_states = self.proj1.forward(&hidden_states)?.apply(&self.act)?;
        self.proj2.forward(&hidden_states)
    }
}

pub(crate) struct Qwen3ASRThinker {
    pub audio_tower: Qwen3ASRAudioEncoder,
    pub model: Qwen3ASRThinkerTextModel,
    pub audio_token_id: u32,
    pub lm_head: candle_nn::Linear,
}

impl Qwen3ASRThinker {
    pub fn new(vb: ShardedVarBuilder, config: &ThinkerConfig) -> Result<Self> {
        let audio_tower = Qwen3ASRAudioEncoder::new(vb.pp("audio_tower"), &config.audio_config)?;
        let model = Qwen3ASRThinkerTextModel::new(vb.pp("model"), &config.text_config)?;
        let lm_head = maybe_lm_head(vb.pp("lm_head"), &config.text_config, &model.embed_tokens)?;
        Ok(Self {
            audio_tower,
            model,
            audio_token_id: config.audio_token_id,
            lm_head,
        })
    }

    pub fn forward_step(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        input_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut input_embeds = self.model.embed_tokens.forward(input_ids)?;
        if let Some(input_features) = input_features {
            let audio_feature = self.audio_tower.forward(input_features)?;
            let audio_mask = get_equal_mask(input_ids, self.audio_token_id)?;
            let n_audio_tokens = audio_mask.sum_all()?.to_scalar::<u32>()?;
            if n_audio_tokens as usize != audio_feature.dim(0)? {
                return Err(candle_core::Error::Msg(format!(
                    "audio token count {} != encoder length {}",
                    n_audio_tokens,
                    audio_feature.dim(0)?
                )));
            }
            input_embeds = masked_scatter_dim0(&input_embeds, &audio_feature, &audio_mask)?;
        }
        let outputs = self.model.forward(&input_embeds, seqlen_offset, None)?;
        self.lm_head.forward(&outputs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
