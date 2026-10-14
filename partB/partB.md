# Part B — Capacity reconciliation

Every number here is printed by `python partB/capacity.py`, which reads `starter_kit/bench/model_spec.md` and
`bench_log.csv`. The full output is saved in `partB/capacity_output.txt`.

---

## B1 — KV cache size and concurrency

**(a) KV-cache bytes per token**

```
2 (K and V) × 28 layers × 8 KV heads × 128 head_dim × 2 bytes (fp16) = 114,688 bytes = 112 KiB per token
```
Only the **8 KV heads** are cached (GQA). The 24 query heads don't store anything. Using 24 would give 344,064 B, which
is 3× too big.

**(b) Maximum concurrent 4096-token sequences**

```
usable GPU memory   = 24 GB × 0.92                = 22.080 GB
− fp16 weights      = 4.2e9 params × 2 B          =  8.400 GB
− runtime overhead  (given)                       =  1.600 GB
= KV pool                                         = 12.080 GB
12.080e9 B / 114,688 B per token                  = 105,329 tokens
105,329 / 4096 tokens per sequence                = 25.7  →  about 25 concurrent sequences
```

**Unit check: GB or GiB?** Reading "24 GB" as GiB gives 118,497 tokens, or 28.9 sequences. I used the log to decide:

**Check against the log.** Implied capacity = tokens resident at peak ÷ `kv_cache_util`:

| row (bs, prompt+gen) | resident tokens | kv_cache_util | implied capacity |
|---|---|---|---|
| 16, 512+256 | 12,288 | 0.12 | 102,400 |
| 64, 512+256 | 49,152 | 0.47 | 104,579 |
| 16, 3584+512 | 65,536 | 0.62 | 105,703 |
| 24, 3584+512 | 98,304 | 0.93 | 105,703 |

Every usable row implies 102k–107k tokens, which matches the decimal-GB prediction (105k), not GiB (118k). Rows with
util of 0.06 or less are too coarsely rounded to use.

**Prediction vs log:**
- Batch 24 × 4096 = 98,304 tokens. Predicted util is 0.933; the log shows **0.93**.
- Batch 32 × 4096 = 131,072 tokens, which is more than 105k, so it can't fit. The log shows the **first preemptions
  (7) exactly at batch 32**.

---

## B2 — The long-context anomaly (prompt 3584, gen 512)

| bs | wall s | goodput (gen tok/s) | ITL p50 ms | TTFT p50 ms | preempted | kv_util | 512×ITL (s) | non-decode time (s) |
|---|---|---|---|---|---|---|---|---|
| 16 | 49.97 | 163.9 | 77.2 | 498 | 0 | 0.62 | 39.5 | 10.4 |
| 24 | 61.16 | **200.9** | 96.1 | 500 | 0 | 0.93 | 49.2 | 12.0 |
| 32 | 94.71 | **173.0** | 101.8 | 637 | **7** | 0.97 | 52.1 | **42.6** |
| 48 | 151.41 | **162.3** | 100.0 | 955 | **23** | 0.97 | 51.2 | **100.2** |

Goodput = num_requests × gen_len / wall_clock_s. Non-decode time = wall − 512 × ITL.

**The anomaly.** Throughput rises with batch up to 24, then *falls*. Batch 48 does 2× the work of batch 24 but takes
2.5× as long.

**Mechanism.**
1. **The KV cache is full.** B1 says about 25.7 sequences of 4096 tokens fit. `kv_cache_util` reaches 0.93 at
   bs 24 and then pins at 0.97.
2. **The scheduler evicts sequences.** `preempted_seqs` goes from 0 to 7 to 23. With vLLM's default recompute
   preemption, an evicted sequence loses its KV and must **re-prefill its 3584-token prompt** later.
3. **Individual decode steps don't get slower.** ITL stays flat at about 100 ms from bs 24 to bs 48. That is
   consistent with only about 25 sequences ever decoding at once, the same as bs 24. The extra batch doesn't add
   parallel work. It adds **waiting and recompute**: non-decode time goes from 12.0 s to 42.6 s to 100.2 s.
4. TTFT goes from 500 to 637 to 955 ms, because new and re-admitted sequences queue for KV blocks.

Up to bs 24, ITL grows linearly with resident KV: ITL ≈ 43.3 ms + 0.568 ms per 1k KV tokens (a fit over the 4 rows
without preemption). The slope implies about 202 GB/s of effective KV-read bandwidth against a 300 GB/s peak. This is
a memory-bandwidth-bound decode, as expected.

**Proposed change: cap concurrency below the KV limit with `max_num_seqs=24`.**

Prediction, built from the log's own measured rows: excess requests wait in a queue instead of thrashing the cache, and
run in waves.

| workload | waves | predicted wall | measured wall | predicted goodput | measured goodput | change |
|---|---|---|---|---|---|---|
| 48 requests | 24 + 24 | 2 × 61.16 = 122.3 s | 151.4 s | **200.9** | 162.3 | **+24%** |
| 32 requests | 24 + 8 | 61.16 + 36.30 = 97.5 s | 94.7 s | 168.1 | 173.0 | −3% |

The cap gives 0 preemptions in both cases. The honest summary: it helps under sustained overload (+24% at 48), and it
is roughly neutral at 32 because the second wave of 8 runs inefficiently.

**Alternative, which needs a quality check:** `kv_cache_dtype=fp8` halves the KV to 57,344 B/token, giving about 51
concurrent 4096-token sequences, so bs 48 fits with no preemption. A rough model (the ITL fit with half the KV slope,
plus the measured prefill rate of about 6.1k tok/s) predicts about **319 gen tok/s at bs 48, roughly 2× today**. That
estimate is less certain: it assumes fp8 KV reads cost half as much and that output quality holds.

---

## B3 — The misread column and honest goodput

**The misread column is `reported_tok_s`.** It counts *prompt + generated* tokens:

```
reported_tok_s = num_requests × (prompt_len + gen_len) / wall_clock_s     ← matches all 13 rows to < 0.5%
```
For example, bs 24 long: 24 × 4096 / 61.16 = **1607.3** (the log says 1607.4). Prompt tokens are input that is
processed in parallel during prefill. They are not output a user receives. Long-prompt rows carry 7× more prompt than
output (3584 vs 512), so they look "fast".

**Goodput of the batch-24 long-prompt row, two independent ways**

1. **From counts and wall clock:** 24 × 512 / 61.16 s = **200.9 generated tok/s**.
2. **From the latency columns:** 24 concurrent sequences ÷ 96.07 ms ITL = **249.8 generated tok/s**.

The two disagree, and the gap tells you something. Method 2 is the steady-state *decode-phase* rate only. Per request,
TTFT + 511 × ITL = 49.6 s, but the run took 61.2 s, so about 11.6 s went to prefilling 86k prompt tokens and ramp-up.
**For capacity planning, use 200.9 tok/s** (end to end). 249.8 tok/s is the ceiling during pure decode.

**Both report conclusions come from that column:**
- "Long prompts give better throughput (1311 vs 883 at bs 16)": in generated tokens, **short prompts give 294.5 vs
  163.9 tok/s, which is 1.80× *better***. The conclusion is reversed.
- "Assume ~1600 tok/s, scale linearly, so bs 48 gives ~3200": 1600 is mostly prompt tokens, and the log **already
  contains a bs-48 row: 1298.5 reported, 162.3 generated tok/s**. Throughput doesn't scale past about 25 sequences
  because of the KV limit (B1, B2).

**What the report should have said:**
> On one L4 with 4096-token requests (3584 in / 512 out), peak output is about 200 generated tok/s, at batch ≈ 24.
> Beyond that the KV cache, which holds about 25 such sequences, forces preemption, and output drops to 162 tok/s at
> batch 48. For short chat (512 in / 256 out), output reaches 756 tok/s at batch 64 with the cache only 47% full.
> Longer prompts *reduce* output throughput. Plan capacity in generated tokens per second, per prompt-length bucket,
> and cap concurrency at about 24 for 4k-context traffic.

---

## B4 — The one metric that confirms the B2 mechanism

I would pull **`vllm:num_requests_running`** (gauge, sampled over each run), with `vllm:num_requests_waiting` alongside
it. The mechanism claims that the KV cache, not the batch size, caps true concurrency. If that is right:
- the running count can never exceed what fits in about 105k KV tokens: about 29 at the start of a long run
  (105,329 / 3,585 tokens per sequence), falling toward **about 25** as sequences grow to 4096;
- for the bs-32 and bs-48 runs it stays well below 32 or 48, with the remainder waiting;
- in the bs-24 run it sits at 24 for the whole run, with 0 waiting;
- after applying `max_num_seqs=24`, it holds at 24 and `vllm:num_preemptions_total` stays at 0.

If instead `num_requests_running` reached 48 in the bs-48 run, my mechanism would be wrong, and the slowdown would have
to come from something else, such as compute saturation.

---

## Data-quality note
`e2e_ms_p95` is *larger* than the whole run's `wall_clock_s` in 12 of 13 rows. For example, bs 1 shows a p95 of
12.86 s for a 10.94 s run. If every request was submitted at t = 0, that is impossible, so the e2e column is measured on
a different clock or definition. I don't use it in any derivation.
