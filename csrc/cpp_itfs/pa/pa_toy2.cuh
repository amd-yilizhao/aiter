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
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel_toy_baseline(
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

    constexpr int HEAD_LOOP          = DIVIDE_ROUND_UP(HEAD_SIZE, 256 / (int)sizeof(scalar_t));
    constexpr int HEAD_SIZE_PER_LOOP = DIVIDE_ROUND_UP(HEAD_SIZE, HEAD_LOOP);
    constexpr int QKHELOOP           = HEAD_SIZE_PER_LOOP / QKHE_PER_FETCH;
    // lane16id 只覆盖 16 槽位；GQA>16 时分多轮。每轮 head 基址步长必须为 16（与 16-lane 分组一致），
    constexpr int GQA_RATIO_LOOP   = DIVIDE_ROUND_UP(GQA_RATIO, 16);
    constexpr int GQA_HEAD_STRIDE = 16;

    constexpr int T_PAR_SIZE      = 256;
    constexpr int NWARPS          = NUM_THREADS / WARP_SIZE;
    constexpr int TOKENS_PER_WARP = T_PAR_SIZE / NWARPS;
    constexpr int TLOOP           = TOKENS_PER_WARP / 16;
    constexpr int vectorize_size  = CONTIGUOUS_KV_ELEMS_16B_LOAD;

    const auto warpid   = thread_idx / WARP_SIZE;
    const auto laneid   = thread_idx % WARP_SIZE;
    const auto rowid    = laneid / 16;
    const auto lane16id = laneid % 16;

    const int context_len = context_lens[seq_idx];
    const int partition_start_token_idx = partition_idx * T_PAR_SIZE;
    if(partition_start_token_idx >= context_len) {
        return;
    }

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int last_ctx_block     = num_context_blocks - 1;

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

    // load q shared mem -> reg（qhead_idx = lane16id + gr * 16）
    _B16x8 Qlocal[GQA_RATIO_LOOP][HEAD_LOOP][QKHELOOP] = {};
    for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
        const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
        if (qhead_idx < GQA_RATIO) {
            for (int head_loop = 0; head_loop < HEAD_LOOP; head_loop++) {
                for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
                    int sh_q_head_element = head_loop * (HEAD_SIZE_PER_LOOP / Q_VEC_SIZE) +
                                            qkhe_depth * ROWS_PER_WARP + rowid;
                    Qlocal[gr][head_loop][qkhe_depth] = shared_q[qhead_idx][sh_q_head_element];
                }
            }
        }
    }

    // load k -> reg && Q*K
    _B16x8 Klocal[HEAD_LOOP][TLOOP][QKHELOOP] = {};
    floatx4 d_out[GQA_RATIO_LOOP][TLOOP] = {};
    float d_out_max[GQA_RATIO_LOOP][TLOOP];
    float score_max[GQA_RATIO_LOOP];
    for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
        score_max[gr] = -FLT_MAX;
        for (int t = 0; t < TLOOP; t++) d_out_max[gr][t] = -FLT_MAX;
    }

    constexpr int K_VEC_SIZE = CONTIGUOUS_KV_ELEMS_16B_LOAD;

    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        const auto local_token_idx = TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
        const auto global_token_idx = partition_start_token_idx + local_token_idx;

        const bool is_valid_token = (global_token_idx < context_len);
        
        const auto block_table_idx = global_token_idx / BLOCK_SIZE;
        const auto block_table_element = global_token_idx % BLOCK_SIZE;

        // 避免越界
        const int safe_block_idx = is_valid_token ? block_table_idx : last_ctx_block;
        const auto block_table_offset = seq_idx * max_num_blocks_per_seq + safe_block_idx;
        const auto physical_block_num = block_tables[block_table_offset];

        for (int head_loop = 0; head_loop < HEAD_LOOP; head_loop++) {
            for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
                // k_cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
                const int head_element = head_loop * HEAD_SIZE_PER_LOOP +
                                         qkhe_depth * QKHE_PER_FETCH + rowid * K_VEC_SIZE;
                Klocal[head_loop][token_depth][qkhe_depth] = *reinterpret_cast<const _B16x8*>(k_cache +
                    physical_block_num * kv_block_stride +
                    kv_head_idx * kv_head_stride +
                    head_element / K_VEC_SIZE * BLOCK_SIZE * K_VEC_SIZE +
                    block_table_element * K_VEC_SIZE);
                for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
                    for (int i = 0; i < 2; i++) {
                        d_out[gr][token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                            Klocal[head_loop][token_depth][qkhe_depth].xy[i],
                            Qlocal[gr][head_loop][qkhe_depth].xy[i],
                            d_out[gr][token_depth]);
                    }
                }
            }
        }
        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            d_out[gr][token_depth] *= scale;
        }

        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            for (int i = 0; i < 4; i++) {
                d_out_max[gr][token_depth] =
                    fmaxf(d_out_max[gr][token_depth], d_out[gr][token_depth][i]);
            }
            for (int mask = QKHE_PER_FETCH; mask >= 16; mask -= 16) {
                d_out_max[gr][token_depth] = fmaxf(d_out_max[gr][token_depth],
                                                   __shfl_xor(d_out_max[gr][token_depth], mask));
            }
            score_max[gr] = fmaxf(score_max[gr], d_out_max[gr][token_depth]);
        }
    }

    __shared__ float shared_score_max[NWARPS][GQA_RATIO];
    __shared__ float shared_score_sum[NWARPS][GQA_RATIO];
    constexpr float k_softmax_sum_eps = 1e-6f;
    __shared__ float shared_score[T_PAR_SIZE][GQA_RATIO];

    if (rowid == 0) {
        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
            if (qhead_idx < GQA_RATIO) {
                shared_score_max[warpid][qhead_idx] = score_max[gr];
            }
        }
    }
    __syncthreads();

    float partition_score_max[GQA_RATIO_LOOP];
    for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) partition_score_max[gr] = -FLT_MAX;
    for (int nwarp = 0; nwarp < NWARPS; nwarp++) {
        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
            if (qhead_idx < GQA_RATIO) {
                partition_score_max[gr] =
                    fmaxf(partition_score_max[gr], shared_score_max[nwarp][qhead_idx]);
            }
        }
    }

    float saved_exp[GQA_RATIO_LOOP][TLOOP][4] = {};
    float score_sum[GQA_RATIO_LOOP][TLOOP]    = {};

    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
            for (int i = 0; i < 4; i++) {
                const auto local_token_idx  = warpid * TOKENS_PER_WARP + token_depth * 16 + rowid * 4 + i;
                const auto global_token_idx = partition_start_token_idx + local_token_idx;
                if (qhead_idx < GQA_RATIO && global_token_idx < context_len) {
                    saved_exp[gr][token_depth][i] =
                        __expf(d_out[gr][token_depth][i] - partition_score_max[gr]);
                    score_sum[gr][token_depth] += saved_exp[gr][token_depth][i];
                } else {
                    saved_exp[gr][token_depth][i] = 0.0f;
                }
            }
            for (int mask = QKHE_PER_FETCH; mask >= 16; mask -= 16) {
                score_sum[gr][token_depth] = score_sum[gr][token_depth] +
                    __shfl_xor(score_sum[gr][token_depth], mask);
            }
            if (rowid == 0 && qhead_idx < GQA_RATIO) {
                if (token_depth == 0)
                    shared_score_sum[warpid][qhead_idx] = score_sum[gr][token_depth];
                else
                    shared_score_sum[warpid][qhead_idx] += score_sum[gr][token_depth];
            }
        }
    }
    __syncthreads();

    float partition_score_sum[GQA_RATIO_LOOP] = {};
    for (int nwarp = 0; nwarp < NWARPS; nwarp++) {
        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
            if (qhead_idx < GQA_RATIO) {
                partition_score_sum[gr] += shared_score_sum[nwarp][qhead_idx];
            }
        }
    }

    float inv_partition_sum[GQA_RATIO_LOOP] = {};
    for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
        const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
        inv_partition_sum[gr] = (qhead_idx < GQA_RATIO)
            ? __fdividef(1.f, partition_score_sum[gr] + k_softmax_sum_eps)
            : 0.f;
    }
    for (int tloop = 0; tloop < TLOOP; tloop++) {
        for (int i = 0; i < 4; i++) {
            const int local_token_idx = warpid * TOKENS_PER_WARP + tloop * 16 + rowid * 4 + i;
            for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
                const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
                if (qhead_idx < GQA_RATIO) {
                    shared_score[local_token_idx][qhead_idx] =
                        saved_exp[gr][tloop][i] * inv_partition_sum[gr];
                }
            }
        }
    }
    __syncthreads();

    if ((warpid == 0) && (rowid == 0)) {
        int num_heads = gridDim.z * GQA_RATIO;
        int max_num_partitions = gridDim.y;
        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
            if (qhead_idx < GQA_RATIO) {
                int global_head_idx = kv_head_idx * GQA_RATIO + qhead_idx;
                int offset          = seq_idx * num_heads * max_num_partitions +
                             global_head_idx * max_num_partitions + partition_idx;
                exp_sums[offset]    = partition_score_sum[gr];
                max_logits[offset]  = partition_score_max[gr];
            }
        }
    }

    //Load V to reg && PV MFMA
    constexpr auto VHELOOP = HEAD_SIZE / 16 / NWARPS;
    constexpr auto VTLOOP = T_PAR_SIZE / TOKENS_PER_WARP;
    constexpr auto VTLANELOOP = TOKENS_PER_WARP / ROWS_PER_WARP / CONTIGUOUS_KV_ELEMS_16B_LOAD;

    constexpr int VTOKENS_PER_LANE = TOKENS_PER_WARP / ROWS_PER_WARP;

    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
        _B16x8 Vlocal[VTLOOP][VTLANELOOP];
        const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;

        floatx4 tmp_out[GQA_RATIO_LOOP] = {};
        for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
            for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
                const auto vlocal_token_idx = vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP +
                                              rowid * VTOKENS_PER_LANE + vfetch_depth * 8;
                const auto vglobal_token_idx = partition_start_token_idx + vlocal_token_idx;

                const auto block_table_idx     = vglobal_token_idx / BLOCK_SIZE;
                const auto block_table_element = vglobal_token_idx % BLOCK_SIZE;

                const int safe_block_idx =
                    (vglobal_token_idx < context_len) ? block_table_idx : last_ctx_block;

                const auto block_table_offset = seq_idx * max_num_blocks_per_seq + safe_block_idx;
                const auto v_physical_block_num = block_tables[block_table_offset];

                const cache_t* v_ptr = v_cache +
                                       v_physical_block_num * kv_block_stride +
                                       kv_head_idx * kv_head_stride +
                                       block_table_element / vectorize_size * HEAD_SIZE * vectorize_size +
                                       vhead_elem * vectorize_size;

                Vlocal[vtoken_depth][vfetch_depth] = *reinterpret_cast<const _B16x8*>(v_ptr);

                for (int i = 0; i < 2; i++) {
                    const auto p_offset = vtoken_depth * TOKENS_PER_WARP + rowid * VTOKENS_PER_LANE +
                                          vfetch_depth * 8 + i * 4;
                    for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
                        const int qhead_idx = gr * GQA_HEAD_STRIDE + lane16id;
                        floatx4 p_floats = (qhead_idx < GQA_RATIO) ? floatx4{shared_score[p_offset + 0][qhead_idx],
                                                                             shared_score[p_offset + 1][qhead_idx],
                                                                             shared_score[p_offset + 2][qhead_idx],
                                                                             shared_score[p_offset + 3][qhead_idx]}
                                                                   : floatx4{0.f, 0.f, 0.f, 0.f};
                        const auto p_weight = from_floatx4<scalar_t>(p_floats);
                        tmp_out[gr] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                            Vlocal[vtoken_depth][vfetch_depth].xy[i], p_weight, tmp_out[gr]);
                    }
                }
            }
        }
        int num_heads = gridDim.z * GQA_RATIO;
        int max_num_partitions = gridDim.y;
        for (int gr = 0; gr < GQA_RATIO_LOOP; gr++) {
            const int qhead_idx = lane16id + gr * GQA_HEAD_STRIDE;
            if (qhead_idx < GQA_RATIO) {
                int global_head_idx = kv_head_idx * GQA_RATIO + qhead_idx;
                for (int idx = 0; idx < 4; idx++) {
                    const auto o_head_elem = vhe_depth * NWARPS * 16 + warpid * 16 + rowid * 4 + idx;
                    // out[num_seqs, num_heads, max_num_partitions, head_size]
                    auto out_offset   = seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                                        global_head_idx * max_num_partitions * HEAD_SIZE +
                                        partition_idx * HEAD_SIZE + o_head_elem;
                    out[out_offset] = static_cast<output_t>(tmp_out[gr][idx]);
                }
            }
        }
    }
}


// Grid: (num_heads, num_seqs)
// Block: (NUM_THREADS)
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS, int PARTITION_SIZE>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel_toy_baseline(
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

            result += weight * (float)tmp_out[tmp_out_value_offset];
        }
        int out_offset = seq_idx * gridDim.x * HEAD_SIZE + head_idx * HEAD_SIZE + head_element;
        out[out_offset] = static_cast<OUTT>(result);
    }

}


