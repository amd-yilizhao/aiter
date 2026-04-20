import math
import torch
from csrc.cpp_itfs.pa.pa_toy import paged_attention_toy

torch.set_default_device("cuda")

PARTITION_SIZE = 256
DTYPE = torch.bfloat16


def torch_reference(
    query,
    k_cache,
    v_cache_shuffle,
    block_tables,
    context_lens,
    num_kv_heads,
    head_size,
    scale,
    block_size: int,
):
    """与 kernel 一致的 V shuffle 布局下的 torch 参考实现。"""
    num_seqs, num_heads, _ = query.shape
    x = 16 // DTYPE.itemsize
    out_ref = torch.zeros(num_seqs, num_heads, head_size, dtype=torch.float32)

    for s in range(num_seqs):
        ctx = context_lens[s].item()
        block_idx_list = block_tables[s, : math.ceil(ctx / block_size)].tolist()

        k_list, v_list = [], []
        for b_idx in block_idx_list:
            k_block = k_cache[b_idx].permute(2, 0, 1, 3)
            k_block = k_block.reshape(block_size, num_kv_heads, head_size)
            k_list.append(k_block)
            v_tokens = [
                v_cache_shuffle[b_idx, :, t // x, :, t % x] for t in range(block_size)
            ]
            v_block = torch.stack(v_tokens, dim=0)
            v_list.append(v_block)

        k_all = torch.cat(k_list, dim=0)[:ctx].float()
        v_all = torch.cat(v_list, dim=0)[:ctx].float()
        gqa = num_heads // num_kv_heads
        k_all = k_all.repeat_interleave(gqa, dim=1)
        v_all = v_all.repeat_interleave(gqa, dim=1)
        q = query[s].float()
        scores = torch.einsum("hd,thd->ht", q, k_all) * scale
        weights = torch.softmax(scores, dim=-1)
        out_ref[s] = torch.einsum("ht,thd->hd", weights, v_all)

    return out_ref.to(DTYPE)


def run_case(
    name: str,
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    ctx_len: int,
    seed: int = 0,
    atol: float = 1e-2,
):
    torch.manual_seed(seed)
    assert num_heads % num_kv_heads == 0

    scale = 1.0 / math.sqrt(head_size)
    x = 16 // DTYPE.itemsize

    blocks_per_seq = math.ceil(ctx_len / block_size)
    num_blocks = num_seqs * blocks_per_seq + 8

    query = torch.randn(num_seqs, num_heads, head_size, dtype=DTYPE)
    k_cache = torch.randn(num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=DTYPE)
    v_cache_shuffle = torch.randn(
        num_blocks, num_kv_heads, block_size // x, head_size, x, dtype=DTYPE
    )
    context_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    max_context_len = ctx_len

    block_tables = torch.zeros(num_seqs, blocks_per_seq, dtype=torch.int32)
    for i in range(num_seqs):
        block_tables[i] = torch.arange(
            i * blocks_per_seq, (i + 1) * blocks_per_seq, dtype=torch.int32
        )

    max_num_partitions = math.ceil(max_context_len / PARTITION_SIZE)
    exp_sums = torch.zeros(num_seqs, num_heads, max_num_partitions)
    max_logits = torch.full((num_seqs, num_heads, max_num_partitions), -float("inf"))
    tmp_out = torch.zeros(num_seqs, num_heads, max_num_partitions, head_size, dtype=DTYPE)
    out_toy = torch.zeros(num_seqs, num_heads, head_size, dtype=DTYPE)

    out_ref = torch_reference(
        query,
        k_cache,
        v_cache_shuffle,
        block_tables,
        context_lens,
        num_kv_heads,
        head_size,
        scale,
        block_size,
    )

    paged_attention_toy(
        out_toy,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        k_cache,
        v_cache_shuffle,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        max_context_len,
    )

    torch.cuda.synchronize()
    max_diff = (out_toy.float() - out_ref.float()).abs().max().item()
    mean_diff = (out_toy.float() - out_ref.float()).abs().mean().item()
    ok = torch.allclose(out_toy.float(), out_ref.float(), atol=atol)
    print(f"[{name}] max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} allclose(atol={atol})={ok}")
    return ok


if __name__ == "__main__":
    BLOCK_SIZE = 16
    all_ok = True

    # 基线：GQA=8, head=128
    all_ok &= run_case(
        "baseline GQA=8 head=128",
        num_seqs=4,
        num_heads=8,
        num_kv_heads=1,
        head_size=128,
        block_size=BLOCK_SIZE,
        ctx_len=200,
        seed=0,
    )

    # GQA_RATIO > 16（触发 GQA_RATIO_LOOP>1 的 else 分支）
    all_ok &= run_case(
        "GQA=24 head=128",
        num_seqs=2,
        num_heads=24,
        num_kv_heads=1,
        head_size=128,
        block_size=BLOCK_SIZE,
        ctx_len=200,
        seed=1,
    )

    # HEAD_SIZE > 256（HEAD_LOOP>1）
    all_ok &= run_case(
        "GQA=8 head=384",
        num_seqs=2,
        num_heads=8,
        num_kv_heads=1,
        head_size=384,
        block_size=BLOCK_SIZE,
        ctx_len=200,
        seed=2,
    )

    if not all_ok:
        raise SystemExit(1)
