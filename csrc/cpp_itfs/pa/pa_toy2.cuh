#pragma once

#include "dtype_fp8.cuh"
#include "float.h"
#include "hip_compat.h"
#include "pa_common.cuh"
#include "pa_kernels.cuh"
#include "quant_utils.cuh"
#include <algorithm>
#include <hip/hip_bf16.h>
#include <type_traits>

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
#define __HIP__GFX9__
#endif


#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))

// grid (num_seqs, max_num_partitions, num_kv_heads)
// block (256)
// clang-format off
template <typename scalar_t, typename output_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO, int MTP=1, vllm::Fp8QuantMethod QUANT_METHOD=vllm::Fp8QuantMethod::kPerTensor, bool V_SHUFFLE=false>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel_toy(
    const scalar_t* __restrict__ q,         // [num_seqs*mtp, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, block_size/x, head_size, x]
    const float scale,
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,       // max_num_blocks_per_seq = block_tables.size(1)
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,                     // q_stride = query.stride(0)
    const int kv_block_stride,              // kv_block_stride = key_cache.stride(0)
    const int kv_head_stride,               // kv_head_stride = key_cache.stride(1)
    float* __restrict__ exp_sums,           // [num_seqs*mtp, num_heads, max_num_partitions]
    float* __restrict__ max_logits,         // [num_seqs*mtp, num_heads, max_num_partitions]
    output_t* __restrict__ out,             // [num_seqs*mtp, num_heads, max_num_partitions, head_size]
    const float* q_scale_ptr,               // [num_seqs*mtp, num_heads]
    const float* k_scale_ptr, const float* v_scale_ptr) {
    
    const auto seq_idx = blockIdx.x;
    const auto partition_idx = blockIdx.y;
    const auto kv_head_idx = blockIdx.z;
    const auto thread_idx = threadIdx.x;
    
    const int context_len = context_lens[seq_idx];
    constexpr int T_PAR_SIZE = 256;
    const int partition_start_token_idx = partition_idx * T_PAR_SIZE;
    if(partition_start_token_idx >= context_len) {
        return;
    }

    // load q from global to shared memory
    __shared__ scalar_t shared_q[GQA_RATIO][HEAD_SIZE];
    constexpr int shared_mem = GQA_RATIO * HEAD_SIZE;

    for (int i = thread_idx; i < shared_mem; i += NUM_THREADS) {
        const int gqa_idx = i / HEAD_SIZE;
        const int q_head_element = i % HEAD_SIZE;
        const int global_q_head_idx = kv_head_idx * GQA_RATIO + gqa_idx;

        shared_q[gqa_idx][q_head_element] = q[seq_idx * q_stride + global_q_head_idx * HEAD_SIZE + q_head_element];
    }
    __syncthreads();

    // caculate k element
    const auto global_token_idx = partition_start_token_idx + thread_idx;
    const bool is_valid_token = (global_token_idx < context_len);
    
    const auto block_table_idx = global_token_idx / BLOCK_SIZE;
    const auto block_table_element = global_token_idx % BLOCK_SIZE;

    // 避免越界
    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int safe_block_idx = is_valid_token ? block_table_idx : (num_context_blocks - 1);
    const auto block_table_offset = seq_idx * max_num_blocks_per_seq + safe_block_idx;
    const auto physical_block_num = block_tables[block_table_offset];
    
    // Q*K
    float score[GQA_RATIO] = {};
    constexpr int vectorize_size = 16 / sizeof(cache_t);

    if (is_valid_token) {
        for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
            for (int head_element = 0; head_element < HEAD_SIZE; head_element++) {
                scalar_t q_val = shared_q[gqa_idx][head_element];
                // [num_blocks, num_kv_heads, head_size/x, block_size, x]
                const cache_t* k_ptr = k_cache + 
                                        physical_block_num * kv_block_stride +
                                        kv_head_idx * kv_head_stride +
                                        head_element / vectorize_size * BLOCK_SIZE * vectorize_size
                                        + block_table_element * vectorize_size + head_element % vectorize_size;
                score[gqa_idx] += float(q_val) * float(*k_ptr);
            }
            score[gqa_idx] *= scale;
        }
    }

    //softmax 
    // qk score
    __shared__ float shared_score[T_PAR_SIZE][GQA_RATIO];
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
        if (is_valid_token) {
            shared_score[thread_idx][gqa_idx] = score[gqa_idx];
        } else {
            shared_score[thread_idx][gqa_idx] = -FLT_MAX;
        }
    }
    __syncthreads();

    // local_max_score
    float score_max[GQA_RATIO];
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
        score_max[gqa_idx] = -FLT_MAX;
        for (int t = 0; t < T_PAR_SIZE; t++) {
            score_max[gqa_idx] = MAX(score_max[gqa_idx], shared_score[t][gqa_idx]);
        }
    }
    
    //exp(score - max)
    float score_exp[GQA_RATIO];
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
        score_exp[gqa_idx] = __expf(score[gqa_idx] - score_max[gqa_idx]);
        if (is_valid_token) {
            shared_score[thread_idx][gqa_idx] = score_exp[gqa_idx];
        } else {
            shared_score[thread_idx][gqa_idx] = 0.0f;
        }
    }
    __syncthreads();

    //sum_exp(score - max) -> shared memory
    float score_sum[GQA_RATIO] = {};
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
        for (int t = 0; t < T_PAR_SIZE; t++) {
            score_sum[gqa_idx] += shared_score[t][gqa_idx];
        }
    }
    __shared__ float shared_score_sum[GQA_RATIO];
    if (thread_idx == 0) {
        for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
            shared_score_sum[gqa_idx] = score_sum[gqa_idx];
        }
    }
    __syncthreads();

    // local out[GQA_RATIO][HEAD_SIZE]
    for (int i = thread_idx; i < GQA_RATIO * HEAD_SIZE; i += NUM_THREADS) {
        int gqa_idx      = i / HEAD_SIZE;
        int head_element = i % HEAD_SIZE;
        
        float result = 0;
        for (int t = 0; t < T_PAR_SIZE; t++) {
            int token_global_idx = partition_start_token_idx + t;
            // 先检查边界，再读取block_tables
            if (token_global_idx < context_len) {
                int block_idx = token_global_idx / BLOCK_SIZE;
                int block_element = token_global_idx % BLOCK_SIZE;
                int global_block_idx = seq_idx * max_num_blocks_per_seq + block_idx;
                int global_block_num = block_tables[global_block_idx];
                
                float weight = shared_score[t][gqa_idx] / shared_score_sum[gqa_idx];
                // v_cache [num_blocks, num_kv_heads, block_size/x, head_size, x]
                const cache_t* v_ptr = v_cache + 
                                        global_block_num * kv_block_stride +
                                        kv_head_idx * kv_head_stride +
                                        block_element / vectorize_size * HEAD_SIZE * vectorize_size +
                                        head_element * vectorize_size + block_element % vectorize_size;
                result += weight * float(*v_ptr);
            }
        }
        int global_head_idx = kv_head_idx * GQA_RATIO + gqa_idx;
        int num_heads = gridDim.z * GQA_RATIO;
        int max_num_partitions = gridDim.y;
        // out[num_seqs, num_heads, max_num_partitions, head_size]
        auto out_offset = seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                          global_head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE + head_element;
        out[out_offset] = static_cast<output_t>(result);
    }

    // exp_sums [num_seqs, num_heads, max_num_partitions]
    // max_logits [num_seqs, num_heads, max_num_partitions]
    if (thread_idx < GQA_RATIO) {
        int gqa_idx = thread_idx;
        int global_head_idx = kv_head_idx * GQA_RATIO + gqa_idx;

        int num_heads = gridDim.z * GQA_RATIO;
        int max_num_partitions = gridDim.y;

        int offset = seq_idx * num_heads * max_num_partitions +
                global_head_idx * max_num_partitions + partition_idx;
        exp_sums[offset] = score_sum[gqa_idx];
        max_logits[offset] = score_max[gqa_idx];
    }
}


// Grid: (num_heads, num_seqs)
// Block: (NUM_THREADS)
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS, int PARTITION_SIZE>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel_toy(
    OUTT* __restrict__ out,                      // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,          // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,        // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,        // [num_seqs, num_heads, max_num_partitions, head_size] - QKV kernel 的输出
    const int* __restrict__ context_lens,        // [num_seqs]
    const int max_num_partitions) {
    
    const auto head_idx = blockIdx.x;
    const auto seq_idx  = blockIdx.y;
    const auto thread_idx = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    const int num_partitions  = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
    
    int max_logits_base_offset = seq_idx * gridDim.x * max_num_partitions + head_idx * max_num_partitions;

    float global_max = -FLT_MAX;
    float sum_p_adjusted;
    float global_sum = 0;
    for (int i = 0; i < num_partitions; i++) {
        global_max = MAX(global_max, max_logits[max_logits_base_offset + i]);
    }

    for (int i = 0; i < num_partitions; i++) {
        sum_p_adjusted = exp_sums[max_logits_base_offset + i] * __expf(max_logits[max_logits_base_offset + i] - global_max);
        global_sum += sum_p_adjusted;
    }

    for (int head_element = thread_idx; head_element < HEAD_SIZE; head_element += NUM_THREADS) {
        float result = 0;
        for (int num_partition = 0; num_partition < num_partitions; num_partition++) {
            int tmp_out_value_offset = seq_idx * gridDim.x * max_num_partitions * HEAD_SIZE +
                        head_idx * max_num_partitions * HEAD_SIZE + num_partition * HEAD_SIZE + head_element;

            float sum_p = exp_sums[max_logits_base_offset + num_partition] *
                             __expf(max_logits[max_logits_base_offset + num_partition] - global_max);
            float weight = sum_p / global_sum;

            result += weight * tmp_out[tmp_out_value_offset];
        }
        int out_offset = seq_idx * gridDim.x * HEAD_SIZE + head_idx * HEAD_SIZE + head_element;
        out[out_offset] = static_cast<OUTT>(result);
    }

}


