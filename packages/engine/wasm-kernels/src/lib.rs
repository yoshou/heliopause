use core::slice;
use std::alloc::{alloc, dealloc, Layout};

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

const OK: i32 = 0;
const ERR_SHAPE: i32 = 1;
const ERR_HEADS: i32 = 2;
const ERR_TYPE: i32 = 3;

const TYPE_Q4_K: i32 = 1;
const TYPE_Q5_K: i32 = 2;
const TYPE_Q6_K: i32 = 3;
const TYPE_IQ4_XS: i32 = 4;
const TYPE_Q8_0: i32 = 5;
const QK_K: usize = 256;
const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[no_mangle]
pub extern "C" fn hp_alloc(byte_len: usize) -> *mut u8 {
    if byte_len == 0 {
        return core::ptr::null_mut();
    }
    let layout = Layout::from_size_align(byte_len, 16).expect("valid wasm allocation layout");
    unsafe { alloc(layout) }
}

#[no_mangle]
pub unsafe extern "C" fn hp_dealloc(ptr: *mut u8, byte_len: usize) {
    if !ptr.is_null() && byte_len != 0 {
        let layout = Layout::from_size_align(byte_len, 16).expect("valid wasm allocation layout");
        dealloc(ptr, layout);
    }
}

#[no_mangle]
pub unsafe extern "C" fn hp_ssm_conv1d_f32(
    conv_input_ptr: *const f32,
    conv_input_len: usize,
    kernel_ptr: *const f32,
    kernel_len: usize,
    channel_count: usize,
    token_count: usize,
    kernel_size: usize,
    output_ptr: *mut f32,
    output_len: usize,
) -> i32 {
    let input_window = kernel_size.saturating_sub(1).saturating_add(token_count);
    if conv_input_len != input_window.saturating_mul(channel_count)
        || kernel_len != kernel_size.saturating_mul(channel_count)
        || output_len != channel_count.saturating_mul(token_count)
    {
        return ERR_SHAPE;
    }

    let conv_input = slice::from_raw_parts(conv_input_ptr, conv_input_len);
    let kernel = slice::from_raw_parts(kernel_ptr, kernel_len);
    let output = slice::from_raw_parts_mut(output_ptr, output_len);

    for token in 0..token_count {
        for channel in 0..channel_count {
            let input_offset = channel * input_window + token;
            let kernel_offset = channel * kernel_size;
            let mut sum = 0.0_f32;
            for k in 0..kernel_size {
                sum = (sum + conv_input[input_offset + k] * kernel[kernel_offset + k]).round_to_f32();
            }
            output[token * channel_count + channel] = sum;
        }
    }

    OK
}

#[no_mangle]
pub unsafe extern "C" fn hp_gated_delta_net_f32(
    query_ptr: *const f32,
    query_len: usize,
    key_ptr: *const f32,
    key_len: usize,
    value_ptr: *const f32,
    value_len: usize,
    gate_ptr: *const f32,
    gate_len: usize,
    beta_ptr: *const f32,
    beta_len: usize,
    state_ptr: *const f32,
    state_len: usize,
    state_size: usize,
    key_head_count: usize,
    value_head_count: usize,
    token_count: usize,
    output_ptr: *mut f32,
    output_len: usize,
    new_state_ptr: *mut f32,
    new_state_len: usize,
) -> i32 {
    if query_len != token_count.saturating_mul(key_head_count).saturating_mul(state_size)
        || key_len != token_count.saturating_mul(key_head_count).saturating_mul(state_size)
        || value_len != token_count.saturating_mul(value_head_count).saturating_mul(state_size)
        || gate_len != token_count.saturating_mul(value_head_count)
        || beta_len != token_count.saturating_mul(value_head_count)
        || state_len != value_head_count.saturating_mul(state_size).saturating_mul(state_size)
        || output_len != token_count.saturating_mul(value_head_count).saturating_mul(state_size)
        || new_state_len != state_len
    {
        return ERR_SHAPE;
    }
    if key_head_count == 0 || value_head_count % key_head_count != 0 {
        return ERR_HEADS;
    }

    let query = slice::from_raw_parts(query_ptr, query_len);
    let key = slice::from_raw_parts(key_ptr, key_len);
    let value = slice::from_raw_parts(value_ptr, value_len);
    let gate = slice::from_raw_parts(gate_ptr, gate_len);
    let beta = slice::from_raw_parts(beta_ptr, beta_len);
    let state = slice::from_raw_parts(state_ptr, state_len);
    let output = slice::from_raw_parts_mut(output_ptr, output_len);
    let new_state = slice::from_raw_parts_mut(new_state_ptr, new_state_len);

    new_state.copy_from_slice(state);
    let mut delta = vec![0.0_f32; state_size];
    let scale = (1.0_f32 / (state_size as f32).sqrt()).round_to_f32();

    for value_head in 0..value_head_count {
        let key_head = value_head % key_head_count;
        let state_offset = value_head * state_size * state_size;

        for token in 0..token_count {
            let q_offset = (token * key_head_count + key_head) * state_size;
            let k_offset = q_offset;
            let v_offset = (token * value_head_count + value_head) * state_size;
            let gate_value = gate[token * value_head_count + value_head];
            let beta_value = beta[token * value_head_count + value_head];
            let exp_gate = (gate_value as f64).exp() as f32;

            for index in 0..state_size * state_size {
                new_state[state_offset + index] = (new_state[state_offset + index] * exp_gate).round_to_f32();
            }

            for j in 0..state_size {
                let row_offset = state_offset + j * state_size;
                let sum = dot_f32(&new_state[row_offset..row_offset + state_size], &key[k_offset..k_offset + state_size]);
                delta[j] = ((value[v_offset + j] - sum).round_to_f32() * beta_value).round_to_f32();
            }

            for j in 0..state_size {
                let row_offset = state_offset + j * state_size;
                let delta_value = delta[j];
                for i in 0..state_size {
                    new_state[row_offset + i] =
                        (new_state[row_offset + i] + (key[k_offset + i] * delta_value).round_to_f32()).round_to_f32();
                }
            }

            let output_offset = (token * value_head_count + value_head) * state_size;
            for j in 0..state_size {
                let row_offset = state_offset + j * state_size;
                let sum = dot_f32(&new_state[row_offset..row_offset + state_size], &query[q_offset..q_offset + state_size]);
                output[output_offset + j] = (sum * scale).round_to_f32();
            }
        }
    }

    OK
}

#[no_mangle]
pub unsafe extern "C" fn hp_matmul_quantized_f32(
    type_id: i32,
    weight_ptr: *const u8,
    weight_len: usize,
    input_ptr: *const f32,
    input_len: usize,
    input_size: usize,
    row_count: usize,
    column_count: usize,
    output_ptr: *mut f32,
    output_len: usize,
) -> i32 {
    if input_len != input_size.saturating_mul(column_count)
        || output_len != row_count.saturating_mul(column_count)
    {
        return ERR_SHAPE;
    }
    let row_bytes = match quantized_row_bytes(type_id, input_size) {
        Some(value) => value,
        None => return ERR_TYPE,
    };
    if weight_len != row_bytes.saturating_mul(row_count) {
        return ERR_SHAPE;
    }

    let weight = slice::from_raw_parts(weight_ptr, weight_len);
    let input = slice::from_raw_parts(input_ptr, input_len);
    let output = slice::from_raw_parts_mut(output_ptr, output_len);

    for column in 0..column_count {
        let input_column = &input[column * input_size..(column + 1) * input_size];
        if type_id == TYPE_Q8_0 {
            let q8 = quantize_q8_0(input_column);
            for row in 0..row_count {
                let row_offset = row * row_bytes;
                output[column * row_count + row] = vec_dot_q8_0_q8_0(&weight[row_offset..row_offset + row_bytes], &q8);
            }
        } else {
            let q8 = quantize_q8_k(input_column);
            for row in 0..row_count {
                let row_offset = row * row_bytes;
                let row_data = &weight[row_offset..row_offset + row_bytes];
                output[column * row_count + row] = match type_id {
                    TYPE_Q4_K => vec_dot_q4_k_q8_k(row_data, &q8),
                    TYPE_Q5_K => vec_dot_q5_k_q8_k(row_data, &q8),
                    TYPE_Q6_K => vec_dot_q6_k_q8_k(row_data, &q8),
                    TYPE_IQ4_XS => vec_dot_iq4_xs_q8_k(row_data, &q8),
                    _ => return ERR_TYPE,
                };
            }
        }
    }

    OK
}

#[no_mangle]
pub unsafe extern "C" fn hp_gqa_attention_f32(
    query_ptr: *const f32,
    query_len: usize,
    key_ptr: *const f32,
    key_len: usize,
    value_ptr: *const f32,
    value_len: usize,
    mask_ptr: *const f32,
    mask_len: usize,
    head_size: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    token_count: usize,
    key_value_token_count: usize,
    scale: f32,
    value_layout: i32,
    quantize_query_f16: i32,
    output_ptr: *mut f32,
    output_len: usize,
) -> i32 {
    if query_len != token_count.saturating_mul(query_head_count).saturating_mul(head_size)
        || key_len != key_value_token_count.saturating_mul(key_value_head_count).saturating_mul(head_size)
        || value_len != key_value_token_count.saturating_mul(key_value_head_count).saturating_mul(head_size)
        || output_len != token_count.saturating_mul(query_head_count).saturating_mul(head_size)
        || (mask_len != 0 && mask_len != token_count.saturating_mul(key_value_token_count))
    {
        return ERR_SHAPE;
    }
    if key_value_head_count == 0 || query_head_count % key_value_head_count != 0 {
        return ERR_HEADS;
    }

    let query = slice::from_raw_parts(query_ptr, query_len);
    let key = slice::from_raw_parts(key_ptr, key_len);
    let value = slice::from_raw_parts(value_ptr, value_len);
    let mask = if mask_len == 0 {
        &[][..]
    } else {
        slice::from_raw_parts(mask_ptr, mask_len)
    };
    let output = slice::from_raw_parts_mut(output_ptr, output_len);
    let group_size = query_head_count / key_value_head_count;
    let mut scores = vec![0.0_f32; key_value_token_count];

    for token in 0..token_count {
        for q_head in 0..query_head_count {
            let kv_head = q_head / group_size;
            let query_offset = (token * query_head_count + q_head) * head_size;
            let mut max_score = f32::NEG_INFINITY;

            for key_token in 0..key_value_token_count {
                let key_offset = (key_token * key_value_head_count + kv_head) * head_size;
                let mut dot = 0.0_f32;
                for index in 0..head_size {
                    let mut query_value = query[query_offset + index];
                    if quantize_query_f16 != 0 {
                        query_value = float16_to_float32(float32_to_float16(query_value));
                    }
                    dot = (dot + (query_value * key[key_offset + index]).round_to_f32()).round_to_f32();
                }
                let mask_value = if mask_len == 0 {
                    0.0
                } else {
                    mask[token * key_value_token_count + key_token]
                };
                let score = if mask_value == f32::NEG_INFINITY {
                    f32::NEG_INFINITY
                } else {
                    ((dot * scale).round_to_f32() + mask_value).round_to_f32()
                };
                scores[key_token] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut sum = 0.0_f32;
            for key_token in 0..key_value_token_count {
                let score = scores[key_token];
                let probability = if score == f32::NEG_INFINITY {
                    0.0
                } else {
                    ((score - max_score) as f64).exp() as f32
                };
                scores[key_token] = probability;
                sum += probability;
            }

            let output_offset = (token * query_head_count + q_head) * head_size;
            for index in 0..head_size {
                let mut weighted = 0.0_f32;
                for key_token in 0..key_value_token_count {
                    let value_offset = if value_layout == 1 {
                        (index * key_value_head_count + kv_head) * key_value_token_count + key_token
                    } else {
                        (key_token * key_value_head_count + kv_head) * head_size + index
                    };
                    weighted = (weighted + ((scores[key_token] / sum) * value[value_offset]).round_to_f32()).round_to_f32();
                }
                output[output_offset + index] = weighted;
            }
        }
    }

    OK
}

fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        return dot_f32_simd(left, right);
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        dot_f32_scalar(left, right)
    }
}

struct QuantizedQ8K {
    d: Vec<f32>,
    qs: Vec<i8>,
    bsums: Vec<i16>,
}

struct QuantizedQ8_0 {
    d: Vec<f32>,
    qs: Vec<i8>,
}

fn quantized_row_bytes(type_id: i32, elements: usize) -> Option<usize> {
    match type_id {
        TYPE_Q4_K if elements % QK_K == 0 => Some(elements / QK_K * 144),
        TYPE_Q5_K if elements % QK_K == 0 => Some(elements / QK_K * 176),
        TYPE_Q6_K if elements % QK_K == 0 => Some(elements / QK_K * 210),
        TYPE_IQ4_XS if elements % QK_K == 0 => Some(elements / QK_K * 136),
        TYPE_Q8_0 if elements % 32 == 0 => Some(elements / 32 * 34),
        _ => None,
    }
}

fn quantize_q8_k(input: &[f32]) -> QuantizedQ8K {
    let block_count = input.len() / QK_K;
    let mut d = vec![0.0_f32; block_count];
    let mut qs = vec![0_i8; input.len()];
    let mut bsums = vec![0_i16; block_count * (QK_K / 16)];

    for block in 0..block_count {
        let base = block * QK_K;
        let mut max = 0.0_f32;
        let mut amax = 0.0_f32;
        for index in 0..QK_K {
            let value = input[base + index];
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                max = value;
            }
        }
        if amax == 0.0 {
            continue;
        }

        let inverse_scale = -127.0 / max;
        for index in 0..QK_K {
            qs[base + index] = (inverse_scale * input[base + index]).round().min(127.0) as i8;
        }
        for group in 0..QK_K / 16 {
            let mut sum = 0_i16;
            for index in 0..16 {
                sum += qs[base + group * 16 + index] as i16;
            }
            bsums[block * (QK_K / 16) + group] = sum;
        }
        d[block] = 1.0 / inverse_scale;
    }

    QuantizedQ8K { d, qs, bsums }
}

fn quantize_q8_0(input: &[f32]) -> QuantizedQ8_0 {
    let block_count = input.len() / 32;
    let mut d = vec![0.0_f32; block_count];
    let mut qs = vec![0_i8; input.len()];

    for block in 0..block_count {
        let base = block * 32;
        let mut amax = 0.0_f32;
        for index in 0..32 {
            amax = amax.max(input[base + index].abs());
        }
        let scale = float16_to_float32(float32_to_float16(amax / 127.0));
        let inverse_scale = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        d[block] = scale;
        for index in 0..32 {
            qs[base + index] = (input[base + index] * inverse_scale).round() as i8;
        }
    }

    QuantizedQ8_0 { d, qs }
}

fn vec_dot_q8_0_q8_0(q8_bytes: &[u8], input: &QuantizedQ8_0) -> f32 {
    let mut sum = 0.0_f32;
    for block in 0..input.d.len() {
        let offset = block * 34;
        let mut isum = 0_i32;
        for index in 0..32 {
            isum += signed_byte(q8_bytes[offset + 2 + index]) as i32 * input.qs[block * 32 + index] as i32;
        }
        sum += isum as f32 * (float16_to_float32(read_u16_le(q8_bytes, offset)) * input.d[block]);
    }
    sum
}

fn vec_dot_q6_k_q8_k(q6_bytes: &[u8], q8: &QuantizedQ8K) -> f32 {
    let mut sums = [0.0_f32; 8];
    let mut aux8 = [0_i8; QK_K];

    for block in 0..q8.d.len() {
        let offset = block * 210;
        let ql = &q6_bytes[offset..offset + 128];
        let qh = &q6_bytes[offset + 128..offset + 192];
        let scales = &q6_bytes[offset + 192..offset + 208];
        let mut ql_offset = 0;
        let mut qh_offset = 0;
        let mut aux_offset = 0;
        let mut aux32 = [0_i32; 8];

        for _group in (0..QK_K).step_by(128) {
            for lane in 0..32 {
                let qh_byte = qh[qh_offset + lane];
                aux8[aux_offset + lane] = (((ql[ql_offset + lane] & 0x0f) | (((qh_byte >> 0) & 3) << 4)) as i16 - 32) as i8;
                aux8[aux_offset + lane + 32] = (((ql[ql_offset + lane + 32] & 0x0f) | (((qh_byte >> 2) & 3) << 4)) as i16 - 32) as i8;
                aux8[aux_offset + lane + 64] = (((ql[ql_offset + lane] >> 4) | (((qh_byte >> 4) & 3) << 4)) as i16 - 32) as i8;
                aux8[aux_offset + lane + 96] = (((ql[ql_offset + lane + 32] >> 4) | (((qh_byte >> 6) & 3) << 4)) as i16 - 32) as i8;
            }
            ql_offset += 64;
            qh_offset += 32;
            aux_offset += 128;
        }

        let mut scale_index = 0;
        let mut value_index = 0;
        let q8_base = block * QK_K;
        for _group in 0..QK_K / 16 {
            let scale = signed_byte(scales[scale_index]) as i32;
            scale_index += 1;
            for lane in 0..8 {
                aux32[lane] += scale * q8.qs[q8_base + value_index + lane] as i32 * aux8[value_index + lane] as i32;
            }
            value_index += 8;
            for lane in 0..8 {
                aux32[lane] += scale * q8.qs[q8_base + value_index + lane] as i32 * aux8[value_index + lane] as i32;
            }
            value_index += 8;
        }

        let d = float16_to_float32(read_u16_le(q6_bytes, offset + 208)) * q8.d[block];
        for lane in 0..8 {
            sums[lane] += d * aux32[lane] as f32;
        }
    }

    sums.iter().sum()
}

fn vec_dot_q5_k_q8_k(q5_bytes: &[u8], q8: &QuantizedQ8K) -> f32 {
    let mut sums = [0.0_f32; 8];
    let mut aux8 = [0_i8; QK_K];
    let mut sumf = 0.0_f32;

    for block in 0..q8.d.len() {
        let offset = block * 176;
        let scales_min_bytes = &q5_bytes[offset + 4..offset + 16];
        let qh = &q5_bytes[offset + 16..offset + 48];
        let qs = &q5_bytes[offset + 48..offset + 176];
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for index in 0..8 {
            let (scale, min) = get_scale_min_k4(index, scales_min_bytes);
            scales[index] = scale;
            mins[index] = min;
        }

        let mut q_offset = 0;
        let mut out = 0;
        let mut high_mask = 1_u8;
        for _group in 0..QK_K / 64 {
            for lane in 0..32 {
                aux8[out + lane] = ((qs[q_offset + lane] & 0x0f) + if qh[lane] & high_mask != 0 { 16 } else { 0 }) as i8;
            }
            out += 32;
            high_mask <<= 1;
            for lane in 0..32 {
                aux8[out + lane] = ((qs[q_offset + lane] >> 4) + if qh[lane] & high_mask != 0 { 16 } else { 0 }) as i8;
            }
            out += 32;
            high_mask <<= 1;
            q_offset += 32;
        }

        let mut sumi = 0_i32;
        let q8_block_offset = block * (QK_K / 16);
        for group in 0..QK_K / 16 {
            sumi += q8.bsums[q8_block_offset + group] as i32 * mins[group / 2] as i32;
        }

        let mut aux32 = [0_i32; 8];
        let mut value_index = 0;
        let q8_base = block * QK_K;
        for group in 0..QK_K / 32 {
            let scale = scales[group] as i32;
            for _chunk in 0..4 {
                for lane in 0..8 {
                    aux32[lane] += scale * q8.qs[q8_base + value_index + lane] as i32 * aux8[value_index + lane] as i32;
                }
                value_index += 8;
            }
        }

        let d = float16_to_float32(read_u16_le(q5_bytes, offset)) * q8.d[block];
        for lane in 0..8 {
            sums[lane] += d * aux32[lane] as f32;
        }
        let dmin = float16_to_float32(read_u16_le(q5_bytes, offset + 2)) * q8.d[block];
        sumf -= dmin * sumi as f32;
    }

    sumf + sums.iter().sum::<f32>()
}

fn vec_dot_q4_k_q8_k(q4_bytes: &[u8], q8: &QuantizedQ8K) -> f32 {
    let mut sums = [0.0_f32; 8];
    let mut aux8 = [0_i8; QK_K];
    let mut sumf = 0.0_f32;

    for block in 0..q8.d.len() {
        let offset = block * 144;
        let scales_min_bytes = &q4_bytes[offset + 4..offset + 16];
        let qs = &q4_bytes[offset + 16..offset + 144];
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for index in 0..8 {
            let (scale, min) = get_scale_min_k4(index, scales_min_bytes);
            scales[index] = scale;
            mins[index] = min;
        }

        let mut q_offset = 0;
        let mut out = 0;
        for _group in 0..QK_K / 64 {
            for lane in 0..32 {
                aux8[out + lane] = (qs[q_offset + lane] & 0x0f) as i8;
            }
            out += 32;
            for lane in 0..32 {
                aux8[out + lane] = (qs[q_offset + lane] >> 4) as i8;
            }
            out += 32;
            q_offset += 32;
        }

        let mut sumi = 0_i32;
        let q8_block_offset = block * (QK_K / 16);
        for group in 0..QK_K / 16 {
            sumi += q8.bsums[q8_block_offset + group] as i32 * mins[group / 2] as i32;
        }

        let mut aux32 = [0_i32; 8];
        let mut value_index = 0;
        let q8_base = block * QK_K;
        for group in 0..QK_K / 32 {
            let scale = scales[group] as i32;
            for _chunk in 0..4 {
                for lane in 0..8 {
                    aux32[lane] += scale * q8.qs[q8_base + value_index + lane] as i32 * aux8[value_index + lane] as i32;
                }
                value_index += 8;
            }
        }

        let d = float16_to_float32(read_u16_le(q4_bytes, offset)) * q8.d[block];
        for lane in 0..8 {
            sums[lane] += d * aux32[lane] as f32;
        }
        let dmin = float16_to_float32(read_u16_le(q4_bytes, offset + 2)) * q8.d[block];
        sumf -= dmin * sumi as f32;
    }

    sumf + sums.iter().sum::<f32>()
}

fn vec_dot_iq4_xs_q8_k(iq4_bytes: &[u8], q8: &QuantizedQ8K) -> f32 {
    let mut accum = [0.0_f32; 8];

    for block in 0..q8.d.len() {
        let offset = block * 136;
        let d4d8 = float16_to_float32(read_u16_le(iq4_bytes, offset)) * q8.d[block];
        let mut scales_h = read_u16_le(iq4_bytes, offset + 2);
        let scales_l = &iq4_bytes[offset + 4..offset + 8];
        let qs = &iq4_bytes[offset + 8..offset + 136];
        let q8_base = block * QK_K;
        let mut qs_offset = 0;
        let mut q8_offset = 0;
        let mut lane_sums = [0_i32; 8];

        for ib in (0..QK_K / 32).step_by(2) {
            let packed_scale = scales_l[ib / 2];
            let ls1 = ((packed_scale & 0x0f) | ((scales_h << 4) as u8 & 0x30)) as i32 - 32;
            let ls2 = ((packed_scale >> 4) | ((scales_h << 2) as u8 & 0x30)) as i32 - 32;
            scales_h >>= 4;

            for j in 0..32 {
                let packed = qs[qs_offset + (j & 15)];
                let q4 = if j < 16 {
                    KVALUES_IQ4_NL[(packed & 0x0f) as usize]
                } else {
                    KVALUES_IQ4_NL[(packed >> 4) as usize]
                } as i32;
                lane_sums[j >> 2] += ls1 * q4 * q8.qs[q8_base + q8_offset + j] as i32;
            }
            qs_offset += 16;
            q8_offset += 32;

            for j in 0..32 {
                let packed = qs[qs_offset + (j & 15)];
                let q4 = if j < 16 {
                    KVALUES_IQ4_NL[(packed & 0x0f) as usize]
                } else {
                    KVALUES_IQ4_NL[(packed >> 4) as usize]
                } as i32;
                lane_sums[j >> 2] += ls2 * q4 * q8.qs[q8_base + q8_offset + j] as i32;
            }
            qs_offset += 16;
            q8_offset += 32;
        }

        for lane in 0..8 {
            accum[lane] = ((d4d8 * lane_sums[lane] as f32).round_to_f32() + accum[lane]).round_to_f32();
        }
    }

    accum.iter().sum()
}

fn get_scale_min_k4(index: usize, q: &[u8]) -> (u8, u8) {
    if index < 4 {
        (q[index] & 63, q[index + 4] & 63)
    } else {
        (
            (q[index + 4] & 0x0f) | ((q[index - 4] >> 6) << 4),
            (q[index + 4] >> 4) | ((q[index] >> 6) << 4),
        )
    }
}

fn signed_byte(value: u8) -> i8 {
    value as i8
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn float16_to_float32(value: u16) -> f32 {
    let sign = if value & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = (value >> 10) & 0x1f;
    let fraction = value & 0x03ff;
    if exponent == 0 {
        return sign * 2.0_f32.powi(-14) * (fraction as f32 / 1024.0);
    }
    if exponent == 31 {
        return if fraction == 0 { sign * f32::INFINITY } else { f32::NAN };
    }
    sign * 2.0_f32.powi(exponent as i32 - 15) * (1.0 + fraction as f32 / 1024.0)
}

fn float32_to_float16(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7e00;
    }
    if value == f32::INFINITY {
        return 0x7c00;
    }
    if value == f32::NEG_INFINITY {
        return 0xfc00;
    }
    let sign = if value < 0.0 || value.is_sign_negative() && value == 0.0 { 0x8000 } else { 0 };
    let abs = value.abs();
    if abs == 0.0 {
        return sign;
    }
    if abs >= 65504.0 {
        return sign | 0x7bff;
    }
    if abs < 2.0_f32.powi(-24) {
        return sign;
    }
    let mut exponent = abs.log2().floor() as i32;
    if exponent < -14 {
        return sign | (abs / 2.0_f32.powi(-24)).round() as u16;
    }
    let mantissa = abs / 2.0_f32.powi(exponent) - 1.0;
    let mut half_mantissa = (mantissa * 1024.0).round() as u16;
    if half_mantissa == 1024 {
        exponent += 1;
        half_mantissa = 0;
    }
    sign | (((exponent + 15) as u16) << 10) | half_mantissa
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn dot_f32_simd(left: &[f32], right: &[f32]) -> f32 {
    let mut index = 0;
    let mut sum = 0.0_f32;
    while index + 4 <= left.len() {
        let a = v128_load(left.as_ptr().add(index) as *const v128);
        let b = v128_load(right.as_ptr().add(index) as *const v128);
        let products = f32x4_mul(a, b);
        let p0 = f32x4_extract_lane::<0>(products);
        let p1 = f32x4_extract_lane::<1>(products);
        let p2 = f32x4_extract_lane::<2>(products);
        let p3 = f32x4_extract_lane::<3>(products);
        sum = (sum + p0).round_to_f32();
        sum = (sum + p1).round_to_f32();
        sum = (sum + p2).round_to_f32();
        sum = (sum + p3).round_to_f32();
        index += 4;
    }

    while index < left.len() {
        sum = (sum + (left[index] * right[index]).round_to_f32()).round_to_f32();
        index += 1;
    }
    sum
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn dot_f32_scalar(left: &[f32], right: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for index in 0..left.len() {
        sum = (sum + (left[index] * right[index]).round_to_f32()).round_to_f32();
    }
    sum
}

trait RoundToF32 {
    fn round_to_f32(self) -> f32;
}

impl RoundToF32 for f32 {
    #[inline]
    fn round_to_f32(self) -> f32 {
        self
    }
}
