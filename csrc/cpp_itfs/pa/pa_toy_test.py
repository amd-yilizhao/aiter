import math
import torch
from csrc.cpp_itfs.pa.pa_toy import paged_attention_toy

torch.set_default_device("cuda")
torch.manual_seed(0)

# ---------- config ----------
NUM_SEQS    = 4
NUM_HEADS   = 8       # query heads
NUM_KV_HEADS = 1      # kv heads → GQA_RATIO = 8
HEAD_SIZE   = 128
BLOCK_SIZE  = 16
CTX_LEN     = 200     # context length per seq (≤ 256 for single partition)
DTYPE       = torch.bfloat16
PARTITION_SIZE = 256
# ----------------------------

scale = 1.0 / math.sqrt(HEAD_SIZE)
x = 16 // DTYPE.itemsize  # = 8 for bf16

num_blocks = math.ceil(NUM_SEQS * CTX_LEN / BLOCK_SIZE) + 4  # some slack

# --- tensors ---
query = torch.randn(NUM_SEQS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)

# k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
k_cache = torch.randn(num_blocks, NUM_KV_HEADS, HEAD_SIZE // x, BLOCK_SIZE, x, dtype=DTYPE)

# v_cache shuffled: [num_blocks, num_kv_heads, block_size/x, head_size, x]
v_cache_shuffle = torch.randn(num_blocks, NUM_KV_HEADS, BLOCK_SIZE // x, HEAD_SIZE, x, dtype=DTYPE)

context_lens = torch.full((NUM_SEQS,), CTX_LEN, dtype=torch.int32)
max_context_len = CTX_LEN

blocks_per_seq = math.ceil(CTX_LEN / BLOCK_SIZE)
block_tables = torch.zeros(NUM_SEQS, blocks_per_seq, dtype=torch.int32)
for i in range(NUM_SEQS):
    block_tables[i] = torch.arange(i * blocks_per_seq, (i + 1) * blocks_per_seq, dtype=torch.int32)

max_num_partitions = math.ceil(max_context_len / PARTITION_SIZE)

# workspace
exp_sums  = torch.zeros(NUM_SEQS, NUM_HEADS, max_num_partitions)
max_logits = torch.full((NUM_SEQS, NUM_HEADS, max_num_partitions), -float("inf"))
tmp_out   = torch.zeros(NUM_SEQS, NUM_HEADS, max_num_partitions, HEAD_SIZE, dtype=DTYPE)
out_toy   = torch.zeros(NUM_SEQS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)

# --- torch reference ---
# V shuffle layout matches kernel: [blocks, kv_heads, block_size/x, head_size, x]
# Logical V[b, kv, h, t] = v_shuffle[b, kv, t // x, h, t % x] — NOT recoverable by
# permute(...).view(...) on the 5D tensor (that interleaves token chunks incorrectly).

out_ref = torch.zeros(NUM_SEQS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)

for s in range(NUM_SEQS):
    ctx = context_lens[s].item()
    # gather K and V for this sequence
    block_idx_list = block_tables[s, : math.ceil(ctx / BLOCK_SIZE)].tolist()

    k_list, v_list = [], []
    for b_idx in block_idx_list:
        # k_cache [blocks, kv_heads, head//x, bs, x] → [bs, kv_heads, head]
        k_block = k_cache[b_idx].permute(2, 0, 1, 3)  # [bs, kv_heads, head//x, x]
        k_block = k_block.reshape(BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
        k_list.append(k_block)
        # [bs, kv_heads, head] from shuffle: V[t, kv, h] = v_shuffle[b, kv, t//x, h, t%x]
        v_tokens = [
            v_cache_shuffle[b_idx, :, t // x, :, t % x] for t in range(BLOCK_SIZE)
        ]
        v_block = torch.stack(v_tokens, dim=0)  # [bs, kv_heads, head]
        v_list.append(v_block)

    k_all = torch.cat(k_list, dim=0)[:ctx].float()  # [ctx, kv_heads, head]
    v_all = torch.cat(v_list, dim=0)[:ctx].float()

    # expand kv heads to match query heads (GQA)
    gqa = NUM_HEADS // NUM_KV_HEADS
    k_all = k_all.repeat_interleave(gqa, dim=1)  # [ctx, num_heads, head]
    v_all = v_all.repeat_interleave(gqa, dim=1)

    q = query[s].float()  # [num_heads, head]

    # scores: [num_heads, ctx]
    scores = torch.einsum("hd,thd->ht", q, k_all) * scale
    weights = torch.softmax(scores, dim=-1)
    out_ref[s] = torch.einsum("ht,thd->hd", weights, v_all)

out_ref = out_ref.to(DTYPE)

# --- run toy kernel ---
paged_attention_toy(
    out_toy, exp_sums, max_logits, tmp_out,
    query, k_cache, v_cache_shuffle,
    NUM_KV_HEADS, scale, block_tables, context_lens, max_context_len,
)

# --- compare ---
torch.cuda.synchronize()
max_diff = (out_toy.float() - out_ref.float()).abs().max().item()
mean_diff = (out_toy.float() - out_ref.float()).abs().mean().item()
print(f"max_diff:  {max_diff:.6f}")
print(f"mean_diff: {mean_diff:.6f}")
print(f"allclose (atol=1e-2): {torch.allclose(out_toy.float(), out_ref.float(), atol=1e-2)}")
