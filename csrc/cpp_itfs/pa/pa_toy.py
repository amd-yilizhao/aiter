import ctypes
import math

from jinja2 import Template

from csrc.cpp_itfs.utils import AITER_CORE_DIR, compile_template_op

MD_NAME = "pa_toy"

with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_toy.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(
    gqa_ratio: int,
    head_size: int,
    dtype: str,
    kv_dtype: str,
    out_dtype: str,
    block_size: int,
    folder: str = None,
):
    return compile_template_op(
        src_template,
        MD_NAME,
        [
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/utils.h",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_toy.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_common.cuh",
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa/pa_kernels.cuh",
            f"{AITER_CORE_DIR}/csrc/include",
            f"{AITER_CORE_DIR}/csrc/include/ck_tile",
        ],
        gqa_ratio=gqa_ratio,
        head_size=head_size,
        dtype=dtype,
        kv_dtype=kv_dtype,
        out_dtype=out_dtype,
        block_size=block_size,
        folder=folder,
    )


def paged_attention_toy(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    max_context_len,
    partition_size=256,
):
    import torch
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    dtype_map = {
        torch.bfloat16: "__hip_bfloat16",
        torch.float16:  "_Float16",
    }

    dtype     = dtype_map[query.dtype]
    kv_dtype  = dtype_map[key_cache.dtype]
    out_dtype = dtype_map[out.dtype]

    num_seqs              = block_tables.size(0)
    num_heads             = query.size(1)
    head_size             = query.size(2)
    q_stride              = query.stride(0)
    max_num_blocks_per_seq = block_tables.size(1)
    kv_block_stride       = key_cache.stride(0)
    kv_head_stride        = key_cache.stride(1)
    gqa_ratio             = int(num_heads / num_kv_heads)
    block_size            = key_cache.size(3)  # k_cache: [blocks, kv_heads, head//x, bs, x]

    # value_cache must be shuffled: [num_blocks, num_kv_heads, block_size/x, head_size, x]
    assert value_cache.dim() == 5, \
        "pa_toy requires shuffled V cache: [num_blocks, num_kv_heads, block_size/x, head_size, x]"

    func = compile(gqa_ratio, head_size, dtype, kv_dtype, out_dtype, block_size)

    (
        out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        tmp_out_ptr,
        scale,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_blocks_per_seq,
        max_context_len,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        stream,
    ) = torch_to_c_types(
        out,
        query,
        key_cache,
        value_cache,
        exp_sums,
        max_logits,
        tmp_out,
        scale,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_blocks_per_seq,
        max_context_len,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        torch.cuda.current_stream(query.device),
    )

    context_lens_ptr = ctypes.cast(
        context_lens.data_ptr(), ctypes.POINTER(ctypes.c_int)
    )
    block_tables_ptr = ctypes.cast(
        block_tables.data_ptr(), ctypes.POINTER(ctypes.c_int)
    )

    func(
        out_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        tmp_out_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        scale,
        block_tables_ptr,
        context_lens_ptr,
        max_context_len,
        num_seqs,
        num_kv_heads,
        num_heads,
        max_num_blocks_per_seq,
        q_stride,
        kv_block_stride,
        kv_head_stride,
        stream,
    )
    return out
