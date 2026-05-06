#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Multimodal façade over [`super::inner::Qwen3ASRThinker`].
//
// Audio conditioning tensor convention:
// - Prefer `pixel_values`: `[batch, n_mels, time]` (same carrier field as vision pipelines).
// - Alternatively pass identical tensor via [`Qwen3AsrSpecificArgs::input_features`].

use std::any::Any;
use std::sync::Mutex;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use crate::amoe::AnyMoeBaseModelMixin;
use crate::device_map::DeviceMapper;
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::pipeline::{
    extract_logits,
    text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
    EitherCache, IsqModel, MultimodalModel, NormalCache, NormalLoadingMetadata,
};

use super::config::Qwen3ASRConfig;
use super::inner::Qwen3ASRThinker;

/// Extra multimodal inputs for Qwen3-ASR (mel features on the prompt step).
#[derive(Clone, Default)]
pub struct Qwen3AsrSpecificArgs {
    pub input_features: Option<Tensor>,
}

pub struct Qwen3AsrModel {
    thinker: Mutex<Qwen3ASRThinker>,
    cfg_meta: ModelConfigMetadata,
    cache: EitherCache,
    device: Device,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl Qwen3AsrModel {
    pub fn new(
        cfg: &Qwen3ASRConfig,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> anyhow::Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        let device = normal_loading_metadata.real_device.clone();
        let vb_thinker = mapper.set_nm_device(vb, false).pp("thinker");
        let thinker = Qwen3ASRThinker::new(vb_thinker, &cfg.thinker_config)?;
        let text = &cfg.thinker_config.text_config;
        let cfg_meta = ModelConfigMetadata {
            max_seq_len: text.max_position_embeddings,
            num_layers: text.num_hidden_layers,
            hidden_size: text.hidden_size,
            num_kv_heads: text.num_key_value_heads,
            num_attn_heads: text.num_attention_heads,
            sliding_window: None,
            k_head_dim: text.head_dim,
            v_head_dim: text.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        let cache = EitherCache::Normal(NormalCache::new(
            text.num_hidden_layers,
            text.max_position_embeddings,
        ));
        Ok(Self {
            thinker: Mutex::new(thinker),
            cfg_meta,
            cache,
            device,
            max_seq_len: text.max_position_embeddings,
            mapper,
        })
    }
}

impl MultimodalModel for Qwen3AsrModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let args = model_specific_args
            .downcast_ref::<Qwen3AsrSpecificArgs>()
            .expect("Qwen3AsrSpecificArgs");
        let mel = pixel_values
            .clone()
            .or_else(|| args.input_features.clone())
            .map(|t| if t.rank() == 3 { t.squeeze(0) } else { Ok(t) })
            .transpose()?;

        let mut thinker = self.thinker.lock().expect("thinker lock");
        let offset = seqlen_offsets.first().copied().unwrap_or(0);
        let logits = thinker.forward_step(input_ids, offset, mel.as_ref())?;
        extract_logits(&logits, context_lens)
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Qwen3AsrSpecificArgs::default())
    }

    fn reset_model_specific_state(&self) {
        if let Ok(mut t) = self.thinker.lock() {
            t.clear_kv_cache();
        }
    }

    fn cache(&self) -> &EitherCache {
        &self.cache
    }

    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg_meta
    }
}

impl IsqModel for Qwen3AsrModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(
            &mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
            Option<usize>,
        )>,
        &dyn DeviceMapper,
    ) {
        (vec![], &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        vec![]
    }
}

impl AnyMoeBaseModelMixin for Qwen3AsrModel {}
