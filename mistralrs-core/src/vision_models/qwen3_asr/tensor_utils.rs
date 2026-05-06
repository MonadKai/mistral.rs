//! Tensor helpers ported from the Apache-2.0 `aha` reference (see NOTICE).

use candle_core::{bail, Result};
use candle_core::{shape::Dim, DType, Device, IndexOp, Tensor, D};

pub(crate) fn nonzero_index_vec(mask: &Tensor) -> Result<Vec<u32>> {
    let mut mask = mask.clone();
    if mask.dtype() != DType::U32 {
        mask = mask.to_dtype(DType::U32)?;
    }
    match mask.rank() {
        0 => bail!("nonzero_index_vec: rank must be > 0"),
        1 => {
            let mask_vector = mask.to_vec1::<u32>()?;
            let indices: Vec<u32> = mask_vector
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val != 0 { Some(idx as u32) } else { None })
                .collect();
            Ok(indices)
        }
        _ => bail!("nonzero_index_vec: rank must be 1"),
    }
}

pub(crate) fn nonzero_slice(mask: &Tensor) -> Result<Vec<(usize, usize)>> {
    let mut index_vec = nonzero_index_vec(mask)?;
    match index_vec.len() {
        0 => Ok(vec![]),
        1 => Ok(vec![(index_vec[0] as usize, (index_vec[0] + 1) as usize)]),
        _ => {
            let mut vec_slice = vec![];
            let mut start = index_vec.remove(0);
            let mut last = start;

            for i in index_vec {
                if i == last + 1 {
                    last = i;
                } else {
                    vec_slice.push((start as usize, (last + 1) as usize));
                    start = i;
                    last = i;
                }
            }
            vec_slice.push((start as usize, (last + 1) as usize));
            Ok(vec_slice)
        }
    }
}

pub(crate) fn prepare_causal_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let arange = Tensor::arange(0u32, tgt_len as u32, device)?;
    let arange = arange.unsqueeze(1)?.broadcast_as((tgt_len, tgt_len))?;
    let upper_triangle = arange.t()?.gt(&arange)?;
    let mask = upper_triangle.where_cond(
        &Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as(arange.shape())?,
        &Tensor::new(0f32, device)?.broadcast_as(arange.shape())?,
    )?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(DType::F32)
}

pub(crate) fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
    let mut tensors = Vec::with_capacity(n_rep);
    for _ in 0..n_rep {
        tensors.push(xs.clone());
    }
    let refs: Vec<&Tensor> = tensors.iter().collect();
    Tensor::cat(&refs, 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
}

pub(crate) fn split_tensor<D: Dim>(t: &Tensor, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
    let dim = dim.to_index(t.shape(), "split")?;
    let mut split_res = Vec::new();
    let mut index = 0;
    for split in splits {
        split_res.push(t.narrow(dim, index, *split)?);
        index += *split;
    }
    Ok(split_res)
}

pub(crate) fn split_tensor_with_size<D: Dim>(
    t: &Tensor,
    splits_size: usize,
    dim: D,
) -> Result<Vec<Tensor>> {
    let dim = dim.to_index(t.shape(), "split")?;
    let mut split_res = Vec::new();
    let dim_size = t.dim(dim)?;
    for (i, split) in (0..dim_size).step_by(splits_size).enumerate() {
        let size = splits_size.min(dim_size - i * splits_size);
        split_res.push(t.narrow(dim, split, size)?);
    }
    Ok(split_res)
}

pub(crate) fn get_equal_mask(input_ids: &Tensor, token_ids: u32) -> Result<Tensor> {
    let tok = Tensor::new(vec![token_ids], input_ids.device())?.to_dtype(input_ids.dtype())?;
    input_ids.broadcast_eq(&tok)?.to_dtype(DType::U32)
}

pub(crate) fn masked_scatter_dim0(
    original: &Tensor,
    replace: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    if original.dim(0)? != 1 || mask.dim(0)? != 1 {
        bail!("masked_scatter_dim0 requires batch size 1");
    }
    let mut original = original.squeeze(0)?;
    let mask = mask.squeeze(0)?;
    let slices = nonzero_slice(&mask)?;
    let mut sub_start = 0usize;
    for (start, end) in slices {
        let sub_end = sub_start + (end - start);
        let sub_replace = replace.i((sub_start..sub_end, ..))?;
        original = original.slice_assign(&[(start..end), (0..original.dim(1)?)], &sub_replace)?;
        sub_start = sub_end;
    }
    original.unsqueeze(0)
}

/// CNN downsampling length mapping used when expanding `<|audio_pad|>` placeholders.
pub(crate) fn get_feat_extract_output_lengths(audio_len: usize) -> usize {
    let input_len_leave = audio_len % 100;
    if input_len_leave > 0 {
        let feat_lengths = (input_len_leave - 1) / 2 + 1;
        ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1 + (audio_len / 100) * 13
    } else {
        (audio_len / 100) * 13
    }
}

pub(crate) fn pad_reflect_last_dim(t: &Tensor, pad: (usize, usize)) -> Result<Tensor> {
    let (pad_l, pad_r) = pad;
    let last_dim = t.dim(D::Minus1)?;
    if pad_l >= last_dim || pad_r >= last_dim {
        bail!("pad_reflect_last_dim: pad too large for tensor last dim");
    }
    let mut pad_tensor = t.clone();
    if pad_l > 0 {
        let left = pad_tensor.narrow(D::Minus1, 1, pad_l)?.contiguous()?;
        let last_dim_id = left.rank() - 1;
        let left_flip = left.flip(&[last_dim_id])?;
        pad_tensor = Tensor::cat(&[&left_flip, &pad_tensor], D::Minus1)?;
    }
    if pad_r > 0 {
        let start_i = last_dim - pad_r;
        let right = pad_tensor.narrow(D::Minus1, start_i, pad_r)?.contiguous()?;
        let last_dim_id = right.rank() - 1;
        let right_flip = right.flip(&[last_dim_id])?;
        pad_tensor = Tensor::cat(&[&pad_tensor, &right_flip], D::Minus1)?;
    }
    Ok(pad_tensor)
}
