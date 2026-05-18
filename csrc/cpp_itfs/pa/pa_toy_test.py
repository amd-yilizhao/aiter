import math
import re
from collections import defaultdict

import torch
from csrc.cpp_itfs.pa.pa_toy import paged_attention_toy
from csrc.cpp_itfs.pa.pa import paged_attention_rocm

torch.set_default_device("cuda")

PARTITION_SIZE = 256
DTYPE = torch.bfloat16
WARMUP = 10
TRACE_ITERS = 20


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


def make_inputs(num_seqs, num_heads, num_kv_heads, head_size, block_size, ctx_len, seed=0):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(head_size)
    x = 16 // DTYPE.itemsize
    blocks_per_seq = math.ceil(ctx_len / block_size)
    num_blocks = num_seqs * blocks_per_seq + 8

    query           = torch.randn(num_seqs, num_heads, head_size, dtype=DTYPE)
    k_cache         = torch.randn(num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=DTYPE)
    v_cache_shuffle = torch.randn(num_blocks, num_kv_heads, block_size // x, head_size, x, dtype=DTYPE)
    context_lens    = torch.full((num_seqs,), ctx_len, dtype=torch.int32)
    block_tables    = torch.zeros(num_seqs, blocks_per_seq, dtype=torch.int32)
    for i in range(num_seqs):
        block_tables[i] = torch.arange(i * blocks_per_seq, (i + 1) * blocks_per_seq, dtype=torch.int32)

    max_num_partitions = math.ceil(ctx_len / PARTITION_SIZE)
    exp_sums   = torch.zeros(num_seqs, num_heads, max_num_partitions)
    max_logits = torch.full((num_seqs, num_heads, max_num_partitions), -float("inf"))
    tmp_out    = torch.zeros(num_seqs, num_heads, max_num_partitions, head_size, dtype=DTYPE)

    return dict(
        query=query, k_cache=k_cache, v_cache_shuffle=v_cache_shuffle,
        context_lens=context_lens, block_tables=block_tables,
        max_context_len=ctx_len, scale=scale, num_kv_heads=num_kv_heads,
        exp_sums=exp_sums, max_logits=max_logits, tmp_out=tmp_out,
    )


def call_kernel(fn, inp):
    out = torch.zeros_like(inp["query"])
    exp_sums   = inp["exp_sums"].clone()
    max_logits = inp["max_logits"].clone()
    tmp_out    = inp["tmp_out"].clone()
    fn(
        out, exp_sums, max_logits, tmp_out,
        inp["query"], inp["k_cache"], inp["v_cache_shuffle"],
        inp["num_kv_heads"], inp["scale"],
        inp["block_tables"], inp["context_lens"], inp["max_context_len"],
    )
    return out


def call_pa_rocm(inp):
    out = torch.zeros_like(inp["query"])
    exp_sums   = inp["exp_sums"].clone()
    max_logits = inp["max_logits"].clone()
    tmp_out    = inp["tmp_out"].clone()
    block_size = inp["k_cache"].size(3)
    paged_attention_rocm(
        out, exp_sums, max_logits, tmp_out,
        inp["query"], inp["k_cache"], inp["v_cache_shuffle"],
        inp["num_kv_heads"], inp["scale"],
        inp["block_tables"], inp["context_lens"],
        block_size, inp["max_context_len"],
        alibi_slopes=None,
        kv_cache_dtype="auto",
    )
    return out


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
    inp = make_inputs(num_seqs, num_heads, num_kv_heads, head_size, block_size, ctx_len, seed)

    out_ref = torch_reference(
        inp["query"], inp["k_cache"], inp["v_cache_shuffle"],
        inp["block_tables"], inp["context_lens"],
        num_kv_heads, head_size, inp["scale"], block_size,
    )

    out_toy  = call_kernel(paged_attention_toy, inp)
    out_rocm = call_pa_rocm(inp)
    torch.cuda.synchronize()

    ok1 = torch.allclose(out_toy.float(),  out_ref.float(), atol=atol)
    ok2 = torch.allclose(out_rocm.float(), out_ref.float(), atol=atol)
    print(f"[{name}] pa_toy allclose={ok1}  pa_rocm allclose={ok2}")
    return ok1 and ok2


if __name__ == "__main__":
    BLOCK_SIZE = 16
    all_ok = True

    all_ok &= run_case("GQA=8  head=64",  num_seqs=4, num_heads=8,  num_kv_heads=1, head_size=64,  block_size=BLOCK_SIZE, ctx_len=200, seed=0)
    all_ok &= run_case("GQA=8  head=128", num_seqs=4, num_heads=8,  num_kv_heads=1, head_size=128, block_size=BLOCK_SIZE, ctx_len=200, seed=1)
    all_ok &= run_case("GQA=24 head=128", num_seqs=2, num_heads=24, num_kv_heads=1, head_size=128, block_size=BLOCK_SIZE, ctx_len=200, seed=2)

    if not all_ok:
        raise SystemExit(1)

    bench_configs = [
        dict(name="GQA=8  head=64  ctx=2048", num_seqs=32, num_heads=8,  num_kv_heads=1, head_size=64,  block_size=BLOCK_SIZE, ctx_len=2048),
        dict(name="GQA=8  head=64  ctx=8192", num_seqs=32, num_heads=8,  num_kv_heads=1, head_size=64,  block_size=BLOCK_SIZE, ctx_len=8192),
        dict(name="GQA=8  head=128 ctx=2048", num_seqs=32, num_heads=8,  num_kv_heads=1, head_size=128, block_size=BLOCK_SIZE, ctx_len=2048),
        dict(name="GQA=8  head=128 ctx=8192", num_seqs=32, num_heads=8,  num_kv_heads=1, head_size=128, block_size=BLOCK_SIZE, ctx_len=8192),
        dict(name="GQA=16 head=64  ctx=2048", num_seqs=32, num_heads=16, num_kv_heads=1, head_size=64,  block_size=BLOCK_SIZE, ctx_len=2048),
        dict(name="GQA=16 head=64  ctx=8192", num_seqs=32, num_heads=16, num_kv_heads=1, head_size=64,  block_size=BLOCK_SIZE, ctx_len=8192),
        dict(name="GQA=16 head=128 ctx=2048", num_seqs=32, num_heads=16, num_kv_heads=1, head_size=128, block_size=BLOCK_SIZE, ctx_len=2048),
        dict(name="GQA=16 head=128 ctx=8192", num_seqs=32, num_heads=16, num_kv_heads=1, head_size=128, block_size=BLOCK_SIZE, ctx_len=8192),
    ]

    # warmup all kernels outside the profiler
    print("\nwarming up...")
    for cfg in bench_configs:
        inp = make_inputs(
            cfg["num_seqs"], cfg["num_heads"], cfg["num_kv_heads"],
            cfg["head_size"], cfg["block_size"], cfg["ctx_len"],
        )
        for _ in range(WARMUP):
            call_kernel(paged_attention_toy, inp)
        for _ in range(WARMUP):
            call_pa_rocm(inp)
    torch.cuda.synchronize()

    kernels = [
        ("pa_toy", lambda inp: call_kernel(paged_attention_toy, inp)),
        ("rocm",   call_pa_rocm),
    ]

    print(f"profiling ({TRACE_ITERS} iters each)...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for cfg in bench_configs:
            inp = make_inputs(
                cfg["num_seqs"], cfg["num_heads"], cfg["num_kv_heads"],
                cfg["head_size"], cfg["block_size"], cfg["ctx_len"],
            )
            case_name = cfg["name"]
            for label, fn in kernels:
                for i in range(TRACE_ITERS):
                    with torch.profiler.record_function(f"[{case_name}] {label} iter{i}"):
                        fn(inp)
        torch.cuda.synchronize()

    prof.export_chrome_trace("trace.json")
    print("trace -> trace.json\n")

    # Parse trace.json: for each gpu_user_annotation span, sum the dur of all
    # 'kernel' events that fall within it. This gives pure GPU execution time.
    import json
    with open("trace.json") as f:
        trace_data = json.load(f)

    all_events = trace_data.get("traceEvents", [])
    qkv_events    = [e for e in all_events
                     if e.get("cat") == "kernel"
                     and "paged_attention" in e.get("name", "")
                     and "reduce" not in e.get("name", "")]
    reduce_events = [e for e in all_events
                     if e.get("cat") == "kernel"
                     and "paged_attention" in e.get("name", "")
                     and "reduce" in e.get("name", "")]

    pattern = re.compile(r'^(\[.+?\] \w+) iter\d+$')
    qkv_times    = defaultdict(list)
    reduce_times = defaultdict(list)
    for span in all_events:
        if span.get("cat") != "gpu_user_annotation":
            continue
        m = pattern.match(span.get("name", ""))
        if not m:
            continue
        key = m.group(1)
        ts, end = span["ts"], span["ts"] + span["dur"]
        qkv_times[key].append(
            sum(k["dur"] for k in qkv_events    if k["ts"] >= ts and k["ts"] + k["dur"] <= end)
        )
        reduce_times[key].append(
            sum(k["dur"] for k in reduce_events if k["ts"] >= ts and k["ts"] + k["dur"] <= end)
        )

    label_order = ["pa_toy", "rocm"]
    lw = max(len(l) for l in label_order)

    print(f"{'case':<32}  {'kernel':<{lw}}  {'qkv us':>8}  {'reduce us':>10}  {'total us':>9}  {'vs pa_toy':>9}")
    print("-" * 85)
    for cfg in bench_configs:
        case_name = cfg["name"]
        times = {}
        for label in label_order:
            key = f"[{case_name}] {label}"
            qkv_vals    = qkv_times.get(key, [])
            reduce_vals = reduce_times.get(key, [])
            if qkv_vals and reduce_vals:
                times[label] = (
                    sum(qkv_vals)    / len(qkv_vals),
                    sum(reduce_vals) / len(reduce_vals),
                )

        ref_total = sum(times["pa_toy"]) if "pa_toy" in times else None
        for label in label_order:
            if label not in times:
                continue
            qkv_us, red_us = times[label]
            total_us = qkv_us + red_us
            ratio = f"{total_us / ref_total:.3f}x" if ref_total else "-"
            print(f"  {case_name:<30}  {label:<{lw}}  {qkv_us:>8.1f}  {red_us:>10.1f}  {total_us:>9.1f}  {ratio:>9}")
        print()
