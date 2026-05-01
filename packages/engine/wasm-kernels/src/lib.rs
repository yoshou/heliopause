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
pub unsafe extern "C" fn hp_prepare_quantized_scales_f32(
    type_id: i32,
    weight_ptr: *const u8,
    weight_len: usize,
    input_size: usize,
    row_count: usize,
    scale_ptr: *mut f32,
    scale_len: usize,
) -> i32 {
    let row_bytes = match quantized_row_bytes(type_id, input_size) {
        Some(value) => value,
        None => return ERR_TYPE,
    };
    if weight_len != row_bytes.saturating_mul(row_count) {
        return ERR_SHAPE;
    }
    let scale_values_per_row = match quantized_scale_values_per_row(type_id, input_size) {
        Some(value) => value,
        None => return ERR_TYPE,
    };
    if scale_len != scale_values_per_row.saturating_mul(row_count) {
        return ERR_SHAPE;
    }

    let weight = slice::from_raw_parts(weight_ptr, weight_len);
    let scales = slice::from_raw_parts_mut(scale_ptr, scale_len);
    let block_count = if type_id == TYPE_Q8_0 { input_size / 32 } else { input_size / QK_K };

    for row in 0..row_count {
        let row_offset = row * row_bytes;
        let scale_offset = row * scale_values_per_row;
        for block in 0..block_count {
            match type_id {
                TYPE_Q4_K => {
                    let offset = row_offset + block * 144;
                    scales[scale_offset + block * 2] = float16_to_float32(read_u16_le(weight, offset));
                    scales[scale_offset + block * 2 + 1] = float16_to_float32(read_u16_le(weight, offset + 2));
                }
                TYPE_Q5_K => {
                    let offset = row_offset + block * 176;
                    scales[scale_offset + block * 2] = float16_to_float32(read_u16_le(weight, offset));
                    scales[scale_offset + block * 2 + 1] = float16_to_float32(read_u16_le(weight, offset + 2));
                }
                TYPE_Q6_K => {
                    let offset = row_offset + block * 210;
                    scales[scale_offset + block] = float16_to_float32(read_u16_le(weight, offset + 208));
                }
                TYPE_IQ4_XS => {
                    let offset = row_offset + block * 136;
                    scales[scale_offset + block] = float16_to_float32(read_u16_le(weight, offset));
                }
                TYPE_Q8_0 => {
                    let offset = row_offset + block * 34;
                    scales[scale_offset + block] = float16_to_float32(read_u16_le(weight, offset));
                }
                _ => return ERR_TYPE,
            }
        }
    }

    OK
}

#[no_mangle]
pub unsafe extern "C" fn hp_matmul_quantized_prepared_f32(
    type_id: i32,
    weight_ptr: *const u8,
    weight_len: usize,
    scale_ptr: *const f32,
    scale_len: usize,
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
    let scale_values_per_row = match quantized_scale_values_per_row(type_id, input_size) {
        Some(value) => value,
        None => return ERR_TYPE,
    };
    if scale_len != scale_values_per_row.saturating_mul(row_count) {
        return ERR_SHAPE;
    }

    let weight = slice::from_raw_parts(weight_ptr, weight_len);
    let scales = slice::from_raw_parts(scale_ptr, scale_len);
    let input = slice::from_raw_parts(input_ptr, input_len);
    let output = slice::from_raw_parts_mut(output_ptr, output_len);

    for column in 0..column_count {
        let input_column = &input[column * input_size..(column + 1) * input_size];
        if type_id == TYPE_Q8_0 {
            let q8 = quantize_q8_0(input_column);
            for row in 0..row_count {
                let row_offset = row * row_bytes;
                let scale_offset = row * scale_values_per_row;
                output[column * row_count + row] = vec_dot_q8_0_q8_0_prepared(
                    &weight[row_offset..row_offset + row_bytes],
                    &q8,
                    &scales[scale_offset..scale_offset + scale_values_per_row],
                );
            }
        } else {
            let q8 = quantize_q8_k(input_column);
            for row in 0..row_count {
                let row_offset = row * row_bytes;
                let scale_offset = row * scale_values_per_row;
                let row_data = &weight[row_offset..row_offset + row_bytes];
                let row_scales = &scales[scale_offset..scale_offset + scale_values_per_row];
                output[column * row_count + row] = match type_id {
                    TYPE_Q4_K => vec_dot_q4_k_q8_k_prepared(row_data, &q8, row_scales),
                    TYPE_Q5_K => vec_dot_q5_k_q8_k_prepared(row_data, &q8, row_scales),
                    TYPE_Q6_K => vec_dot_q6_k_q8_k_prepared(row_data, &q8, row_scales),
                    TYPE_IQ4_XS => vec_dot_iq4_xs_q8_k_prepared(row_data, &q8, row_scales),
                    _ => return ERR_TYPE,
                };
            }
        }
    }

    OK
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hp_matmul_quantized_multi_f32(
    count: usize,
    input_ptr: *const f32,
    input_len: usize,
    input_size: usize,
    column_count: usize,
    type_id_0: i32,
    weight_ptr_0: *const u8,
    weight_len_0: usize,
    row_count_0: usize,
    output_ptr_0: *mut f32,
    output_len_0: usize,
    type_id_1: i32,
    weight_ptr_1: *const u8,
    weight_len_1: usize,
    row_count_1: usize,
    output_ptr_1: *mut f32,
    output_len_1: usize,
    type_id_2: i32,
    weight_ptr_2: *const u8,
    weight_len_2: usize,
    row_count_2: usize,
    output_ptr_2: *mut f32,
    output_len_2: usize,
    type_id_3: i32,
    weight_ptr_3: *const u8,
    weight_len_3: usize,
    row_count_3: usize,
    output_ptr_3: *mut f32,
    output_len_3: usize,
) -> i32 {
    if count == 0 || count > 4 || input_len != input_size.saturating_mul(column_count) {
        return ERR_SHAPE;
    }

    let specs = [
        (type_id_0, weight_ptr_0, weight_len_0, row_count_0, output_ptr_0, output_len_0),
        (type_id_1, weight_ptr_1, weight_len_1, row_count_1, output_ptr_1, output_len_1),
        (type_id_2, weight_ptr_2, weight_len_2, row_count_2, output_ptr_2, output_len_2),
        (type_id_3, weight_ptr_3, weight_len_3, row_count_3, output_ptr_3, output_len_3),
    ];

    for (index, (type_id, _weight_ptr, weight_len, row_count, _output_ptr, output_len)) in specs.iter().enumerate() {
        if index >= count {
            break;
        }
        let Some(row_bytes) = quantized_row_bytes(*type_id, input_size) else {
            return ERR_TYPE;
        };
        if *weight_len != row_bytes.saturating_mul(*row_count)
            || *output_len != row_count.saturating_mul(column_count)
        {
            return ERR_SHAPE;
        }
    }

    let input = slice::from_raw_parts(input_ptr, input_len);
    for column in 0..column_count {
        let input_column = &input[column * input_size..(column + 1) * input_size];
        let mut q8_k: Option<QuantizedQ8K> = None;
        let mut q8_0: Option<QuantizedQ8_0> = None;

        for (index, (type_id, weight_ptr, weight_len, row_count, output_ptr, output_len)) in specs.iter().enumerate() {
            if index >= count {
                break;
            }
            let row_bytes = quantized_row_bytes(*type_id, input_size).expect("validated row bytes");
            let weight = slice::from_raw_parts(*weight_ptr, *weight_len);
            let output = slice::from_raw_parts_mut(*output_ptr, *output_len);
            let output_base = column * *row_count;

            if *type_id == TYPE_Q8_0 {
                if q8_0.is_none() {
                    q8_0 = Some(quantize_q8_0(input_column));
                }
                let q8 = q8_0.as_ref().expect("q8_0 initialized");
                for row in 0..*row_count {
                    let row_offset = row * row_bytes;
                    output[output_base + row] =
                        vec_dot_q8_0_q8_0(&weight[row_offset..row_offset + row_bytes], q8);
                }
            } else {
                if q8_k.is_none() {
                    q8_k = Some(quantize_q8_k(input_column));
                }
                let q8 = q8_k.as_ref().expect("q8_k initialized");
                for row in 0..*row_count {
                    let row_offset = row * row_bytes;
                    let row_data = &weight[row_offset..row_offset + row_bytes];
                    output[output_base + row] = match *type_id {
                        TYPE_Q4_K => vec_dot_q4_k_q8_k(row_data, q8),
                        TYPE_Q5_K => vec_dot_q5_k_q8_k(row_data, q8),
                        TYPE_Q6_K => vec_dot_q6_k_q8_k(row_data, q8),
                        TYPE_IQ4_XS => vec_dot_iq4_xs_q8_k(row_data, q8),
                        _ => return ERR_TYPE,
                    };
                }
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
            let mut quantized_query = Vec::new();
            let query_values: &[f32] = if quantize_query_f16 != 0 {
                quantized_query.reserve(head_size);
                for index in 0..head_size {
                    quantized_query.push(float16_to_float32(float32_to_float16(query[query_offset + index])));
                }
                &quantized_query
            } else {
                &query[query_offset..query_offset + head_size]
            };
            let mut max_score = f32::NEG_INFINITY;

            for key_token in 0..key_value_token_count {
                let key_offset = (key_token * key_value_head_count + kv_head) * head_size;
                let mut dot = 0.0_f32;
                for index in 0..head_size {
                    dot = (dot + (query_values[index] * key[key_offset + index]).round_to_f32()).round_to_f32();
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

fn quantized_scale_values_per_row(type_id: i32, elements: usize) -> Option<usize> {
    match type_id {
        TYPE_Q4_K | TYPE_Q5_K if elements % QK_K == 0 => Some(elements / QK_K * 2),
        TYPE_Q6_K | TYPE_IQ4_XS if elements % QK_K == 0 => Some(elements / QK_K),
        TYPE_Q8_0 if elements % 32 == 0 => Some(elements / 32),
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
        let isum = dot_i8_i8_32(&q8_bytes[offset + 2..offset + 34], &input.qs, block * 32);
        sum += isum as f32 * (float16_to_float32(read_u16_le(q8_bytes, offset)) * input.d[block]);
    }
    sum
}

fn vec_dot_q8_0_q8_0_prepared(q8_bytes: &[u8], input: &QuantizedQ8_0, scales: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for block in 0..input.d.len() {
        let offset = block * 34;
        let isum = dot_i8_i8_32(&q8_bytes[offset + 2..offset + 34], &input.qs, block * 32);
        sum += isum as f32 * (scales[block] * input.d[block]);
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
            accumulate_scaled_i8_i8_16_lanes(&mut aux32, scale, &q8.qs, q8_base + value_index, &aux8, value_index);
            value_index += 16;
        }

        let d = float16_to_float32(read_u16_le(q6_bytes, offset + 208)) * q8.d[block];
        for lane in 0..8 {
            sums[lane] += d * aux32[lane] as f32;
        }
    }

    sums.iter().sum()
}

fn vec_dot_q6_k_q8_k_prepared(q6_bytes: &[u8], q8: &QuantizedQ8K, scales_f32: &[f32]) -> f32 {
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
            accumulate_scaled_i8_i8_16_lanes(&mut aux32, scale, &q8.qs, q8_base + value_index, &aux8, value_index);
            value_index += 16;
        }

        let d = scales_f32[block] * q8.d[block];
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
            accumulate_scaled_i8_i8_32_lanes(&mut aux32, scale, &q8.qs, q8_base + value_index, &aux8, value_index);
            value_index += 32;
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

fn vec_dot_q5_k_q8_k_prepared(q5_bytes: &[u8], q8: &QuantizedQ8K, scales_f32: &[f32]) -> f32 {
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
            accumulate_scaled_i8_i8_32_lanes(&mut aux32, scale, &q8.qs, q8_base + value_index, &aux8, value_index);
            value_index += 32;
        }

        let d = scales_f32[block * 2] * q8.d[block];
        for lane in 0..8 {
            sums[lane] += d * aux32[lane] as f32;
        }
        let dmin = scales_f32[block * 2 + 1] * q8.d[block];
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
            accumulate_scaled_i8_i8_32_lanes(&mut aux32, scale, &q8.qs, q8_base + value_index, &aux8, value_index);
            value_index += 32;
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

fn vec_dot_q4_k_q8_k_prepared(q4_bytes: &[u8], q8: &QuantizedQ8K, scales_f32: &[f32]) -> f32 {
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
            accumulate_scaled_i8_i8_32_lanes(&mut aux32, scale, &q8.qs, q8_base + value_index, &aux8, value_index);
            value_index += 32;
        }

        let d = scales_f32[block * 2] * q8.d[block];
        for lane in 0..8 {
            sums[lane] += d * aux32[lane] as f32;
        }
        let dmin = scales_f32[block * 2 + 1] * q8.d[block];
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

            accumulate_iq4_nl_32_quads(&mut lane_sums, ls1, &q8.qs, q8_base + q8_offset, &qs[qs_offset..qs_offset + 16]);
            qs_offset += 16;
            q8_offset += 32;

            accumulate_iq4_nl_32_quads(&mut lane_sums, ls2, &q8.qs, q8_base + q8_offset, &qs[qs_offset..qs_offset + 16]);
            qs_offset += 16;
            q8_offset += 32;
        }

        for lane in 0..8 {
            accum[lane] = ((d4d8 * lane_sums[lane] as f32).round_to_f32() + accum[lane]).round_to_f32();
        }
    }

    accum.iter().sum()
}

fn vec_dot_iq4_xs_q8_k_prepared(iq4_bytes: &[u8], q8: &QuantizedQ8K, scales_f32: &[f32]) -> f32 {
    let mut accum = [0.0_f32; 8];

    for block in 0..q8.d.len() {
        let offset = block * 136;
        let d4d8 = scales_f32[block] * q8.d[block];
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

            accumulate_iq4_nl_32_quads(&mut lane_sums, ls1, &q8.qs, q8_base + q8_offset, &qs[qs_offset..qs_offset + 16]);
            qs_offset += 16;
            q8_offset += 32;

            accumulate_iq4_nl_32_quads(&mut lane_sums, ls2, &q8.qs, q8_base + q8_offset, &qs[qs_offset..qs_offset + 16]);
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

fn accumulate_scaled_i8_i8_16_lanes(
    accum: &mut [i32; 8],
    scale: i32,
    left: &[i8],
    left_offset: usize,
    right: &[i8],
    right_offset: usize,
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        accumulate_scaled_i8_i8_16_lanes_simd(accum, scale, left, left_offset, right, right_offset);
        return;
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        for base in [0, 8] {
            for lane in 0..8 {
                accum[lane] += scale * left[left_offset + base + lane] as i32 * right[right_offset + base + lane] as i32;
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn accumulate_scaled_i8_i8_16_lanes_simd(
    accum: &mut [i32; 8],
    scale: i32,
    left: &[i8],
    left_offset: usize,
    right: &[i8],
    right_offset: usize,
) {
    let left_values = v128_load(left.as_ptr().add(left_offset) as *const v128);
    let right_values = v128_load(right.as_ptr().add(right_offset) as *const v128);
    let products = i16x8_add(
        i16x8_extmul_low_i8x16(left_values, right_values),
        i16x8_extmul_high_i8x16(left_values, right_values),
    );

    accum[0] += scale * i16x8_extract_lane::<0>(products) as i32;
    accum[1] += scale * i16x8_extract_lane::<1>(products) as i32;
    accum[2] += scale * i16x8_extract_lane::<2>(products) as i32;
    accum[3] += scale * i16x8_extract_lane::<3>(products) as i32;
    accum[4] += scale * i16x8_extract_lane::<4>(products) as i32;
    accum[5] += scale * i16x8_extract_lane::<5>(products) as i32;
    accum[6] += scale * i16x8_extract_lane::<6>(products) as i32;
    accum[7] += scale * i16x8_extract_lane::<7>(products) as i32;
}

fn dot_i8_i8_32(left: &[u8], right: &[i8], right_offset: usize) -> i32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        return dot_i8_i8_32_simd(left, right, right_offset);
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let mut sum = 0_i32;
        for index in 0..32 {
            sum += signed_byte(left[index]) as i32 * right[right_offset + index] as i32;
        }
        sum
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn dot_i8_i8_32_simd(left: &[u8], right: &[i8], right_offset: usize) -> i32 {
    let left_a = v128_load(left.as_ptr() as *const v128);
    let right_a = v128_load(right.as_ptr().add(right_offset) as *const v128);
    let left_b = v128_load(left.as_ptr().add(16) as *const v128);
    let right_b = v128_load(right.as_ptr().add(right_offset + 16) as *const v128);
    let products_a = i16x8_add(
        i16x8_extmul_low_i8x16(left_a, right_a),
        i16x8_extmul_high_i8x16(left_a, right_a),
    );
    let products_b = i16x8_add(
        i16x8_extmul_low_i8x16(left_b, right_b),
        i16x8_extmul_high_i8x16(left_b, right_b),
    );
    let sums = i32x4_add(
        i32x4_extadd_pairwise_i16x8(products_a),
        i32x4_extadd_pairwise_i16x8(products_b),
    );
    i32x4_extract_lane::<0>(sums)
        + i32x4_extract_lane::<1>(sums)
        + i32x4_extract_lane::<2>(sums)
        + i32x4_extract_lane::<3>(sums)
}

fn accumulate_iq4_nl_32_quads(
    accum: &mut [i32; 8],
    scale: i32,
    q8: &[i8],
    q8_offset: usize,
    packed: &[u8],
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        accumulate_iq4_nl_32_quads_simd(accum, scale, q8, q8_offset, packed);
        return;
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        for index in 0..32 {
            let byte = packed[index & 15];
            let q4 = if index < 16 {
                KVALUES_IQ4_NL[(byte & 0x0f) as usize]
            } else {
                KVALUES_IQ4_NL[(byte >> 4) as usize]
            };
            accum[index >> 2] += scale * q4 as i32 * q8[q8_offset + index] as i32;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn accumulate_iq4_nl_32_quads_simd(
    accum: &mut [i32; 8],
    scale: i32,
    q8: &[i8],
    q8_offset: usize,
    packed: &[u8],
) {
    let lookup = i8x16(
        KVALUES_IQ4_NL[0],
        KVALUES_IQ4_NL[1],
        KVALUES_IQ4_NL[2],
        KVALUES_IQ4_NL[3],
        KVALUES_IQ4_NL[4],
        KVALUES_IQ4_NL[5],
        KVALUES_IQ4_NL[6],
        KVALUES_IQ4_NL[7],
        KVALUES_IQ4_NL[8],
        KVALUES_IQ4_NL[9],
        KVALUES_IQ4_NL[10],
        KVALUES_IQ4_NL[11],
        KVALUES_IQ4_NL[12],
        KVALUES_IQ4_NL[13],
        KVALUES_IQ4_NL[14],
        KVALUES_IQ4_NL[15],
    );
    let packed_values = v128_load(packed.as_ptr() as *const v128);
    let nibble_mask = u8x16_splat(0x0f);
    let low_values = i8x16_swizzle(lookup, v128_and(packed_values, nibble_mask));
    let high_values = i8x16_swizzle(lookup, v128_and(u8x16_shr(packed_values, 4), nibble_mask));

    accumulate_iq4_nl_16_quads(accum, 0, scale, v128_load(q8.as_ptr().add(q8_offset) as *const v128), low_values);
    accumulate_iq4_nl_16_quads(accum, 4, scale, v128_load(q8.as_ptr().add(q8_offset + 16) as *const v128), high_values);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn accumulate_iq4_nl_16_quads(
    accum: &mut [i32; 8],
    accum_offset: usize,
    scale: i32,
    q8_values: v128,
    q4_values: v128,
) {
    let pairs_low = i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(q4_values, q8_values));
    let pairs_high = i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(q4_values, q8_values));
    accum[accum_offset] += scale * (i32x4_extract_lane::<0>(pairs_low) + i32x4_extract_lane::<1>(pairs_low));
    accum[accum_offset + 1] += scale * (i32x4_extract_lane::<2>(pairs_low) + i32x4_extract_lane::<3>(pairs_low));
    accum[accum_offset + 2] += scale * (i32x4_extract_lane::<0>(pairs_high) + i32x4_extract_lane::<1>(pairs_high));
    accum[accum_offset + 3] += scale * (i32x4_extract_lane::<2>(pairs_high) + i32x4_extract_lane::<3>(pairs_high));
}

fn accumulate_scaled_i8_i8_32_lanes(
    accum: &mut [i32; 8],
    scale: i32,
    left: &[i8],
    left_offset: usize,
    right: &[i8],
    right_offset: usize,
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        accumulate_scaled_i8_i8_32_lanes_simd(accum, scale, left, left_offset, right, right_offset);
        return;
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        for chunk in 0..4 {
            let base = chunk * 8;
            for lane in 0..8 {
                accum[lane] += scale * left[left_offset + base + lane] as i32 * right[right_offset + base + lane] as i32;
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
unsafe fn accumulate_scaled_i8_i8_32_lanes_simd(
    accum: &mut [i32; 8],
    scale: i32,
    left: &[i8],
    left_offset: usize,
    right: &[i8],
    right_offset: usize,
) {
    let left_a = v128_load(left.as_ptr().add(left_offset) as *const v128);
    let right_a = v128_load(right.as_ptr().add(right_offset) as *const v128);
    let left_b = v128_load(left.as_ptr().add(left_offset + 16) as *const v128);
    let right_b = v128_load(right.as_ptr().add(right_offset + 16) as *const v128);

    let products = i16x8_add(
        i16x8_add(
            i16x8_extmul_low_i8x16(left_a, right_a),
            i16x8_extmul_high_i8x16(left_a, right_a),
        ),
        i16x8_add(
            i16x8_extmul_low_i8x16(left_b, right_b),
            i16x8_extmul_high_i8x16(left_b, right_b),
        ),
    );

    accum[0] += scale * i16x8_extract_lane::<0>(products) as i32;
    accum[1] += scale * i16x8_extract_lane::<1>(products) as i32;
    accum[2] += scale * i16x8_extract_lane::<2>(products) as i32;
    accum[3] += scale * i16x8_extract_lane::<3>(products) as i32;
    accum[4] += scale * i16x8_extract_lane::<4>(products) as i32;
    accum[5] += scale * i16x8_extract_lane::<5>(products) as i32;
    accum[6] += scale * i16x8_extract_lane::<6>(products) as i32;
    accum[7] += scale * i16x8_extract_lane::<7>(products) as i32;
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn float16_to_float32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as i32;
    let fraction = (value & 0x03ff) as u32;

    let bits = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let mut mantissa = fraction;
            let mut normalized_exponent = -14_i32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                normalized_exponent -= 1;
            }
            mantissa &= 0x03ff;
            sign | (((normalized_exponent + 127) as u32) << 23) | (mantissa << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | (((exponent + 112) as u32) << 23) | (fraction << 13)
    };

    f32::from_bits(bits)
}

fn float32_to_float16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        return if mantissa == 0 { sign | 0x7c00 } else { 0x7e00 };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7bff;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let rounded = (mantissa_with_hidden + (1_u32 << (shift - 1))) >> shift;
        return sign | rounded as u16;
    }

    let rounded_mantissa = mantissa + 0x0000_1000;
    if rounded_mantissa & 0x0080_0000 != 0 {
        let rounded_exponent = half_exponent + 1;
        if rounded_exponent >= 31 {
            return sign | 0x7bff;
        }
        return sign | ((rounded_exponent as u16) << 10);
    }

    sign | ((half_exponent as u16) << 10) | ((rounded_mantissa >> 13) as u16)
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
