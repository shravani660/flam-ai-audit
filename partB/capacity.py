#!/usr/bin/env python3
"""
capacity.py -- Part B arithmetic, computed from starter_kit/bench/* (nothing typed in by hand
except the spec values, which are parsed from model_spec.md and printed for checking).

Usage:  python partB/capacity.py
"""

import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "starter_kit", "bench")


def spec():
    txt = open(os.path.join(BENCH, "model_spec.md"), encoding="utf-8").read()

    def num(label):
        m = re.search(r"\|\s*" + re.escape(label) + r"[^|]*\|[^|\d]*([\d.]+)", txt)
        return float(m.group(1))

    return dict(
        params=num("parameters") * 1e9,
        layers=int(num("layers")),
        q_heads=int(num("attention heads (Q)")),
        kv_heads=int(num("KV heads (GQA)")),
        head_dim=int(num("head_dim")),
        gpu_gb=float(re.search(r"L4 \((\d+) GB\)", txt).group(1)),
        util=num("`gpu_memory_utilization`"),
        overhead_gb=num("non-KV runtime overhead"),
        max_len=int(num("`max_model_len`")),
        bw=num("memory bandwidth (peak)"),
    )


def log():
    with open(os.path.join(BENCH, "bench_log.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k])
    return rows


def main():
    s = spec()
    rows = log()
    print("spec parsed:", s, "\n")

    # ------------------------------------------------------------ B1
    print("=" * 78, "\nB1  KV cache\n" + "=" * 78)
    bpe = 2  # fp16
    kv_tok = 2 * s["layers"] * s["kv_heads"] * s["head_dim"] * bpe
    print(f"(a) KV bytes/token = 2 (K,V) x {s['layers']} layers x {s['kv_heads']} KV heads x "
          f"{s['head_dim']} head_dim x {bpe} B = {kv_tok:,} B = {kv_tok / 1024:.0f} KiB")
    wrong = 2 * s["layers"] * s["q_heads"] * s["head_dim"] * bpe
    print(f"    (common mistake: using 24 Q heads instead of 8 KV heads -> {wrong:,} B, 3x too big)")

    weights = s["params"] * bpe
    for unit, GB in (("GB = 1e9 B", 1e9), ("GiB = 2^30 B", 2 ** 30)):
        budget = s["gpu_gb"] * GB * s["util"]
        kv_pool = budget - weights - s["overhead_gb"] * GB
        toks = kv_pool / kv_tok
        print(f"\n(b) reading '24 GB' and '1.6 GB' as {unit}:")
        print(f"    usable = {s['gpu_gb']:.0f} x {GB:,.0f} x {s['util']} = {budget / 1e9:.3f}e9 B")
        print(f"    - weights {s['params'] / 1e9:.1f}e9 params x 2 B = {weights / 1e9:.2f}e9 B")
        print(f"    - overhead {s['overhead_gb']} x {GB:,.0f} = {s['overhead_gb'] * GB / 1e9:.3f}e9 B")
        print(f"    = KV pool {kv_pool / 1e9:.3f}e9 B / {kv_tok:,} B = {toks:,.0f} tokens"
              f" / {s['max_len']} = {toks / s['max_len']:.2f} concurrent 4096-token sequences")

    print("\ncheck against the log -- implied KV capacity = tokens resident / kv_cache_util:")
    print(f"{'bs':>4}{'prompt':>7}{'gen':>5}{'resident tok':>14}{'kv_util':>9}{'implied capacity':>18}{'= 4096-seqs':>13}")
    for r in rows:
        resident = r["num_requests"] * (r["prompt_len"] + r["gen_len"])  # peak, all seqs at full length
        if r["preempted_seqs"] == 0 and r["kv_cache_util"] >= 0.1:  # tiny utils are too rounded to use
            cap = resident / r["kv_cache_util"]
            print(f"{r['batch_size']:>4.0f}{r['prompt_len']:>7.0f}{r['gen_len']:>5.0f}{resident:>14,.0f}"
                  f"{r['kv_cache_util']:>9.2f}{cap:>18,.0f}{cap / 4096:>13.1f}")
    print("first preemption appears at batch 32 x 4096 tokens = 131,072 tokens > capacity")

    # ------------------------------------------------------------ B3 (needed for B2)
    print("\n" + "=" * 78, "\nB3  what does reported_tok_s count?\n" + "=" * 78)
    print(f"{'bs':>4}{'prompt':>7}{'reported':>10}{'gen-only':>10}{'(p+g)/wall':>12}{'match':>7}"
          f"{'goodput = n*gen/wall':>22}")
    for r in rows:
        n, w = r["num_requests"], r["wall_clock_s"]
        gen_only = n * r["gen_len"] / w
        total = n * (r["prompt_len"] + r["gen_len"]) / w
        r["goodput"] = gen_only
        ok = abs(total - r["reported_tok_s"]) / r["reported_tok_s"] < 0.005
        print(f"{n:>4.0f}{r['prompt_len']:>7.0f}{r['reported_tok_s']:>10.1f}{gen_only:>10.1f}{total:>12.1f}"
              f"{'yes' if ok else 'NO':>7}{gen_only:>22.1f}")
    print("-> reported_tok_s = num_requests x (prompt_len + gen_len) / wall_clock_s in every row:")
    print("   it counts PROMPT tokens as throughput.")

    b24 = next(r for r in rows if r["batch_size"] == 24 and r["prompt_len"] == 3584)
    g1 = b24["num_requests"] * b24["gen_len"] / b24["wall_clock_s"]
    g2 = b24["batch_size"] * 1000 / b24["itl_ms_p50"]
    per_req = b24["ttft_ms_p50"] / 1000 + (b24["gen_len"] - 1) * b24["itl_ms_p50"] / 1000
    print(f"\nbatch-24 long-prompt goodput:")
    print(f"  way 1 (counts / wall clock): 24 x 512 / {b24['wall_clock_s']} s = {g1:.1f} gen tok/s")
    print(f"  way 2 (latency columns): 24 concurrent seqs / ITL {b24['itl_ms_p50']} ms = {g2:.1f} gen tok/s"
          f" (steady-state decode only)")
    print(f"        per-request time from TTFT + 511 x ITL = {per_req:.2f} s vs wall {b24['wall_clock_s']} s"
          f" -> {b24['wall_clock_s'] - per_req:.1f} s of the run is not decode (prefill queueing/ramp)")
    s16 = next(r for r in rows if r["batch_size"] == 16 and r["prompt_len"] == 512)
    l16 = next(r for r in rows if r["batch_size"] == 16 and r["prompt_len"] == 3584)
    print(f"\nreport's batch-16 comparison: reported 883.2 (short) vs 1311.4 (long) -> 'long is better'")
    print(f"  goodput: short {s16['goodput']:.1f} vs long {l16['goodput']:.1f} gen tok/s"
          f" -> short prompts produce {s16['goodput'] / l16['goodput']:.2f}x MORE output tokens/s")
    b48 = next(r for r in rows if r["batch_size"] == 48)
    print(f"report's 'batch 48 -> ~3200 tok/s': the log HAS a batch-48 row: reported {b48['reported_tok_s']},"
          f" goodput {b48['goodput']:.1f}")

    # ------------------------------------------------------------ B2
    print("\n" + "=" * 78, "\nB2  long-context sweep (prompt 3584)\n" + "=" * 78)
    longs = [r for r in rows if r["prompt_len"] == 3584]
    print(f"{'bs':>4}{'wall s':>8}{'reported':>10}{'goodput':>9}{'gp/seq':>8}{'ttft':>7}{'itl':>7}"
          f"{'e2e95 s':>9}{'preempt':>9}{'kv':>6}{'512*itl s':>11}{'non-decode s':>13}")
    for r in longs:
        dec = r["gen_len"] * r["itl_ms_p50"] / 1000
        print(f"{r['batch_size']:>4.0f}{r['wall_clock_s']:>8.2f}{r['reported_tok_s']:>10.1f}{r['goodput']:>9.1f}"
              f"{r['goodput'] / r['batch_size']:>8.2f}{r['ttft_ms_p50']:>7.0f}{r['itl_ms_p50']:>7.1f}"
              f"{r['e2e_ms_p95'] / 1000:>9.1f}{r['preempted_seqs']:>9.0f}{r['kv_cache_util']:>6.2f}"
              f"{dec:>11.1f}{r['wall_clock_s'] - dec:>13.1f}")

    # ITL vs resident KV tokens for the no-preemption rows (decode reads all resident KV each step)
    pts = [(r["batch_size"] * (r["prompt_len"] + r["gen_len"] / 2), r["itl_ms_p50"]) for r in longs
           if r["preempted_seqs"] == 0]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    icpt = my - slope * mx
    eff_bw = kv_tok / (slope / 1000) / 1e9
    print(f"\nlinear fit, no-preemption long rows: ITL_ms = {icpt:.1f} + {slope * 1000:.3f} x (resident KV k-tokens)")
    print(f"  slope implies each extra KV token costs {slope * 1e6:.0f} ns/step -> {eff_bw:.0f} GB/s effective"
          f" KV read bandwidth (peak {s['bw']:.0f})")
    print(f"  intercept {icpt:.1f} ms vs weights-only floor {weights / (s['bw'] * 1e9) * 1000:.1f} ms "
          f"(8.4 GB / 300 GB/s)")

    # Proposed change: cap concurrency at 24 (max_num_seqs=24) -> no preemption
    print("\nPROPOSED: max_num_seqs=24 (admission cap below the ~25.7-seq KV limit)")
    for r in longs:
        if r["batch_size"] > 24:
            waves = []
            left = r["batch_size"]
            while left > 0:
                waves.append(min(24, left))
                left -= 24
            pred = sum(next(x for x in longs if x["batch_size"] == w)["wall_clock_s"]
                       if any(x["batch_size"] == w for x in longs) else float("nan") for w in waves)
            gp = r["num_requests"] * r["gen_len"] / pred
            print(f"  batch {r['batch_size']:.0f}: waves {waves} -> predicted wall {pred:.1f} s"
                  f" (measured {r['wall_clock_s']}), goodput {gp:.1f} vs {r['goodput']:.1f}"
                  f" ({(gp / r['goodput'] - 1) * 100:+.0f}%), preemptions 0")

    print("\nALTERNATIVE: kv_cache_dtype=fp8 -> KV bytes/token halves")
    kv8 = kv_tok // 2
    pool = s["gpu_gb"] * 1e9 * s["util"] - weights - s["overhead_gb"] * 1e9
    print(f"  {kv8:,} B/token -> {pool / kv8:,.0f} tokens -> {pool / kv8 / 4096:.1f} concurrent 4096-seqs"
          f" (batch 48 fits, 0 preemptions)")
    resident48 = 48 * (3584 + 256)
    itl48 = icpt + slope / 2 * resident48
    prefill_rate = sum(r["batch_size"] * r["prompt_len"] / (r["wall_clock_s"] - r["gen_len"] * r["itl_ms_p50"] / 1000)
                       for r in longs if r["preempted_seqs"] == 0) / n
    wall48 = 48 * 3584 / prefill_rate + 512 * itl48 / 1000
    print(f"  ITL model with half the KV slope: batch 48 ITL ~ {itl48:.0f} ms;"
          f" prefill ~{prefill_rate:,.0f} tok/s (avg of no-preempt rows)")
    print(f"  predicted wall ~ {wall48:.0f} s -> goodput ~ {48 * 512 / wall48:.0f} gen tok/s"
          f" vs measured {b48['goodput']:.0f}  (rough: assumes fp8 KV read cost = half of fp16)")

    # ------------------------------------------------------------ data-quality notes
    print("\n" + "=" * 78, "\ndata-quality checks\n" + "=" * 78)
    bad = [r for r in rows if r["e2e_ms_p95"] / 1000 > r["wall_clock_s"]]
    print(f"e2e_ms_p95 > wall_clock_s in {len(bad)}/{len(rows)} rows (e.g. bs1: e2e p95 "
          f"{rows[0]['e2e_ms_p95'] / 1000:.2f} s > whole run {rows[0]['wall_clock_s']} s).")
    print("  If all requests are submitted at t=0, a request cannot take longer than the run.")
    print("  -> e2e is measured on a different clock (likely client-side incl. connection/queue).")
    print("  I do not use e2e_ms_p95 in any derivation.")


if __name__ == "__main__":
    main()
