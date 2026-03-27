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
    
    
    constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD = 16 / sizeof(cache_t);
    constexpr int ROWS_PER_WARP                = WARP_SIZE / 16;
    constexpr int QKHE_PER_FETCH               = CONTIGUOUS_KV_ELEMS_16B_LOAD * ROWS_PER_WARP;
    constexpr int QKHELOOP                     = HEAD_SIZE / QKHE_PER_FETCH;

    // constexpr int GQA_RATIO_LOOP            = DIVIDE_ROUND_UP(GQA_RATIO, 16);

    constexpr int T_PAR_SIZE      = 256;
    constexpr int NWARPS          = NUM_THREADS / WARP_SIZE;
    constexpr int TOKENS_PER_WARP = T_PAR_SIZE / NWARPS;
    constexpr int TLOOP           = TOKENS_PER_WARP / 16;

    const auto warpid   = thread_idx / WARP_SIZE;
    const auto laneid   = thread_idx % WARP_SIZE;
    const auto rowid    = laneid / 16;
    const auto lane16id = laneid % 16;

    const int context_len = context_lens[seq_idx];
    const int partition_start_token_idx = partition_idx * T_PAR_SIZE;
    if(partition_start_token_idx >= context_len) {
        return;
    }

    // load q from global to shared memory
    constexpr int Q_VEC_SIZE = 16 / sizeof(scalar_t);
    __shared__ _B16x8 shared_q[GQA_RATIO][HEAD_SIZE / Q_VEC_SIZE];
    constexpr int shared_mem = GQA_RATIO * HEAD_SIZE / Q_VEC_SIZE;

    for (int i = thread_idx; i < shared_mem; i += NUM_THREADS) {
        const int gqa_idx = i / (HEAD_SIZE / Q_VEC_SIZE);
        const int q_head_element = i % (HEAD_SIZE / Q_VEC_SIZE);
        const int global_q_head_idx = kv_head_idx * GQA_RATIO + gqa_idx;

        shared_q[gqa_idx][q_head_element] = *reinterpret_cast<const _B16x8*>(
                q + seq_idx * q_stride + global_q_head_idx * HEAD_SIZE + q_head_element * Q_VEC_SIZE);
    }
    __syncthreads();

    // load q shared mem -> reg
    _B16x8 Qlocal[QKHELOOP] = {};
    if (lane16id < GQA_RATIO) {
        for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
            int sh_q_head_element = qkhe_depth * (QKHE_PER_FETCH / CONTIGUOUS_KV_ELEMS_16B_LOAD) + rowid;
            Qlocal[qkhe_depth] = shared_q[lane16id][sh_q_head_element];
        }
    }

    // load k -> reg && Q*K
    _B16x8 Klocal[TLOOP][QKHELOOP] = {};
    floatx4 d_out[TLOOP] = {};
    constexpr int K_VEC_SIZE = CONTIGUOUS_KV_ELEMS_16B_LOAD;

    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        const auto local_token_idx = TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
        const auto global_token_idx = partition_start_token_idx + local_token_idx;

        const bool is_valid_token = (global_token_idx < context_len);
        
        const auto block_table_idx = global_token_idx / BLOCK_SIZE;
        const auto block_table_element = global_token_idx % BLOCK_SIZE;

        // 避免越界
        const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
        const int safe_block_idx = is_valid_token ? block_table_idx : (num_context_blocks - 1);
        const auto block_table_offset = seq_idx * max_num_blocks_per_seq + safe_block_idx;
        const auto physical_block_num = block_tables[block_table_offset];

        for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
            // k_cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
            const int head_element = qkhe_depth * QKHE_PER_FETCH + rowid * K_VEC_SIZE;
            Klocal[token_depth][qkhe_depth] = *reinterpret_cast<const _B16x8*>(k_cache +
                                                        physical_block_num * kv_block_stride +
                                                        kv_head_idx * kv_head_stride +
                                                        head_element / K_VEC_SIZE * BLOCK_SIZE * K_VEC_SIZE +
                                                        block_table_element * K_VEC_SIZE);
            for (int i = 0; i < 2; i++) {
                d_out[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                    Klocal[token_depth][qkhe_depth].xy[i], Qlocal[qkhe_depth].xy[i], d_out[token_depth]);
            }
        }
        d_out[token_depth] *= scale;
    }


    //softmax 
    // qk score
    __shared__ float shared_score[T_PAR_SIZE][GQA_RATIO];
    __shared__ float shared_score_max[NUM_THREADS / WARP_SIZE][GQA_RATIO];
    __shared__ float shared_score_sum[NUM_THREADS / WARP_SIZE][GQA_RATIO];

    // local_max_score
    float score_max[GQA_RATIO];
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
        if (is_valid_token) {
            score_max[gqa_idx] = score[gqa_idx];
        } else {
            score_max[gqa_idx] = -FLT_MAX;
        }

        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
            score_max[gqa_idx] = fmaxf(score_max[gqa_idx], __shfl_xor(score_max[gqa_idx], mask));
        }
        if (laneid == 0) {
            shared_score_max[warpid][gqa_idx] = score_max[gqa_idx];
        }
        __syncthreads();
        for (int i = 0; i < (NUM_THREADS / WARP_SIZE); i++){
            score_max[gqa_idx] = fmaxf(score_max[gqa_idx], shared_score_max[i][gqa_idx]);
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
            score_exp[gqa_idx] = 0.0f;
        }
    }
    __syncthreads();

    // sum_exp(score - max) -> shared memory
    float score_sum[GQA_RATIO] = {};
    for (int gqa_idx = 0; gqa_idx < GQA_RATIO; gqa_idx++) {
        score_sum[gqa_idx] = 0.0f;
        score_sum[gqa_idx] += score_exp[gqa_idx];
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
            score_sum[gqa_idx] =  score_sum[gqa_idx] + __shfl_xor(score_sum[gqa_idx], mask);
        }
        if (laneid == 0) {
            shared_score_sum[warpid][gqa_idx] = score_sum[gqa_idx];
        }
        __syncthreads();
        for (int i = 0; i < (NUM_THREADS / WARP_SIZE); i++) {
            if (i != warpid) {
                score_sum[gqa_idx] += shared_score_sum[i][gqa_idx];
            }
        }
    }

    for (int i = thread_idx; i < GQA_RATIO * HEAD_SIZE; i += NUM_THREADS) {
        int gqa_idx      = i / HEAD_SIZE;
        int head_element = i % HEAD_SIZE;
        
        float result = 0;
        for (int t = 0; t < T_PAR_SIZE; t += vectorize_size) {
            int token_global_idx = partition_start_token_idx + t;
            // 先检查边界，再读取block_tables
            float weight[vectorize_size] = {};
            for (int s = 0; s < vectorize_size; s++){
                weight[s] = shared_score[t + s][gqa_idx] / score_sum[gqa_idx];
            }
            if (token_global_idx < context_len) {
                int block_idx = token_global_idx / BLOCK_SIZE;
                int block_element = token_global_idx % BLOCK_SIZE;
                int global_block_idx = seq_idx * max_num_blocks_per_seq + block_idx;
                int global_block_num = block_tables[global_block_idx];

                // v_cache [num_blocks, num_kv_heads, block_size/x, head_size, x]
                const cache_t* v_ptr = v_cache + 
                                        global_block_num * kv_block_stride +
                                        kv_head_idx * kv_head_stride +
                                        block_element / vectorize_size * HEAD_SIZE * vectorize_size +
                                        head_element * vectorize_size;
                const _B16x8 v_value = *reinterpret_cast<const _B16x8*>(v_ptr);
                for (int k = 0; k < 2; k++) {
                    for (int j = 0; j < 4; j++) {
                        result += weight[k*4+j] * float(v_value.xy[k][j]);
                    }
                }
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


