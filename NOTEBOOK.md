# Lab notebook

> **Provenance (see AI_USAGE.md):** Entries 1–8 record my AI-assisted working sessions: I used Claude Code to run commands and draft analysis, directing every step. The order, dead ends and corrections are exactly as they happened. Entries from 9 onward are my own independent re-runs and checks.

Chronological. Each entry: hypothesis → experiment → result → what changed in my thinking.
Commands are run from `submission/` unless stated. Python = local venv (3.12, tiktoken 0.14, transformers 5.19).

---

## Entry 1 — Reproduce the baseline before touching anything

**Hypothesis:** the report numbers came from this exact script; if I can't reproduce them, everything downstream is moot.

**Command:**
```
cd starter_kit
python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
```

**Result:** eng 1.27 / 0.226, hin 7.45 / 1.579, ratio 5.89×. Matches REPORT_v0 exactly.

**So what:** the script is the source of the numbers. Any flaw I find in it propagates directly into the report.

**First read of the data (before any code):** the files are described as "parallel line-by-line", but
eng line 1 = "Bengaluru International Airport handled record traffic in March." and hin line 1 = "मुझे सुबह की चाय बहुत पसंद है।" ("I like morning tea very much"). These are not translations of each other. Need to check every line → Entry 3.

---

## Entry 2 — Build the A1 corpus (FLORES-200)

**Why FLORES-200:** truly parallel (same 2009 sentences translated by professionals into every language), open licence (CC-BY-SA 4.0), has all languages I need. Downloaded the original tarball from `dl.fbaipublicfiles.com` (no HF login needed; the HF mirrors `facebook/flores` and `openlanguagedata/flores_plus` are gated).

**Command:** `python partA/get_corpus.py`

**Result:** 2009 sentences (dev 997 + devtest 1012) × 8 languages (eng, hin, kan, tam, tel, mal + ben, mar for Part C). Line counts identical → parallel check passes. Domains: wikinews 689, wikivoyage 668, wikibooks 652.

**Surprises:**
- First extraction attempt failed: archive member names start with `./` → matched by suffix instead. (Dead end, 5 min.)
- NFC is **not** a no-op on real data: it changed 183/2009 Hindi lines and 1195/2009 Bengali lines. I had assumed FLORES was already NFC. Must measure whether this changes token counts before deciding the script's NFC line is harmless.
- Kannada line 2 has a leading space and runs of 2–3 spaces. The `split(" ")` issue will matter more on real data than on the sample.

---

## Entry 3 — Ablation: flip one suspected flaw at a time (gpt2)

**Method:** `partA/audit_ablation.py` re-implements `analyze()` with each questionable choice behind a switch. Control: the default configuration is asserted equal to the imported original `analyze()` → "control OK" on both corpora, so any delta below is caused only by the flipped switch.

**Commands:**
```
python partA/audit_ablation.py --pair sample
python partA/audit_ablation.py --pair flores
```

**Results (hin/eng tokens-per-word ratio; baseline sample 5.89, FLORES 6.10):**

| switch flipped | sample ratio | FLORES ratio |
|---|---|---|
| no lowercasing | 6.06 (+2.9%) | 6.32 (+3.6%) |
| split() not split(" ") | 5.92 (+0.6%) | 6.11 (+0.0%) |
| ratio of totals | 5.91 (+0.3%) | 6.12 (+0.2%) |
| no NFC | 5.89 (0%) | 6.10 (0%) ← invalid test, see below |
| tok/char with graphemes | tok/char ratio 7.00 → 10.86 | 7.12 → 10.89 |
| tok/char with UTF-8 bytes | 7.00 → 2.65 | 7.12 → 2.77 |

**I was wrong about lowercasing direction.** My hypothesis was "lowercasing makes English cheaper, so it flatters English". Measured: the opposite. Lowercasing *raises* English gpt2 tokens/word 1.229 → 1.265 (gpt2's BPE learned capitalised forms like "The", "NASA" as whole tokens; lowercase versions split more). Hindi is untouched (no case). So `.lower()` inflates the English baseline and makes Hindi look ~3–4% *less* bad than it is. Still a bug (asymmetric preprocessing that alters only one side), but the direction is the reverse of what I'd have written without measuring.

**split(" "):** real but tiny on eng/hin. Need per-language check — Kannada has many multi-space runs.

**mean-of-lines vs ratio-of-totals:** real but ~0.2–0.3%. Will report it as a correctness issue with small measured magnitude, not as a headline.

**tok/char:** the "per character" ratio is 2.65×, 7.00× or 10.86× depending only on what you call a character. This kills report finding 2 ("tok/char agrees, so the result is robust"): both metrics share the same numerator, and the per-char ratio is mostly a statement about the denominator.

**Dead end — NFC test invalid:** "no NFC" shows 0% on FLORES, but `get_corpus.py` already wrote NFC text, so switching it off in the reader changes nothing. Need to test on the raw tarball text → Entry 5.

---

## Entry 4 — Corrected benchmark on FLORES, 4 tokenizers, 4 denominators

**Hypothesis (from the report):** "Hindi has more Unicode characters per word, so *any* tokenizer will struggle — property of the script."
**Prediction if true:** an Indic-aware tokenizer should show a similar Hindi/English gap to gpt2.

**Command:**
```
python partA/fertility_v1.py --tokenizer gpt2 --tokenizer hf:unsloth/Llama-3.2-1B \
  --tokenizer hf:sarvamai/sarvam-1 --tokenizer hf:FacebookAI/xlm-roberta-base \
  --langs eng,hin,kan,tam,tel,mal,ben,mar --out partA/results/main
```

Tokenizer choice: gpt2 (what v0 used), Llama-3.2 (128,256 vocab — chosen as a *proxy* for our 128k-vocab FLM-4B, whose tokenizer is not in the kit), sarvam-1 (Indic-focused LLM tokenizer), XLM-R (multilingual, 250k).

**Result — premium vs English, tokens per parallel sentence (Hindi):** gpt2 7.44×, Llama-3.2 2.53×, XLM-R 1.25×, sarvam-1 1.14×.
Same sentences, same script, 6.5× spread between tokenizers → **the report's root-cause claim is refuted**: it is the tokenizer, not the script.

**Surprise 1 — per-word distorts in OPPOSITE directions by language.** Words per parallel sentence: eng 21.3, hin 25.0, kan 15.7, tam 16.4, tel 16.6, mal 14.6.
Hindi writes postpositions as separate words (में, को, है) → more words → per-word ratio *understates* Hindi (gpt2: 6.34× per word vs 7.44× per sentence).
Dravidian languages are agglutinative → fewer, longer words → per-word ratio *overstates* them (gpt2 Malayalam: 22.15× per word vs 15.15× per sentence).
This is the conceptual flaw: "tokens per word" holds the *word* constant, but a word is not the same amount of meaning across languages. Only the parallel sentence holds content constant.

**Surprise 2 — gpt2 is at ~1 token per UTF-8 byte for all 4 Dravidian languages** (0.979–0.996). It has essentially no merges for these scripts; it is a byte-level fallback. Hindi is at 0.595 (some Devanagari merges exist).

**Note to self:** sarvam-1 costs *more* on English (30.5 tok/sent) than gpt2/Llama (26.3/26.4). Routing all traffic to an Indic tokenizer is not free for English.

UNK rate is 0.00% for all four tokenizers, so no tokenizer is "winning" by dropping text to [UNK].

---

## Entry 5 — The remaining suspects, one experiment each (`partA/audit_extras.py`)

| exp | result | verdict |
|---|---|---|
| `spaces` | empty "words" from `split(" ")`: kan 1069 (+3.39% words), tel +1.35%, mal +0.69%, hin +0.03%, eng 1 | bug; tiny for eng/hin, material for Kannada |
| `lower` | lowercasing adds gpt2 tokens to 1629 English word occurrences, removes in 214. "European" 1→3, "September" 1→3. In the sample: 3 lines (NASA/ISRO, GPU, Quarterly Review) | confirms Entry 3 mechanism |
| `bootstrap` | v0 ratio on 10 sample lines: 5.89× with 95% CI [5.37, 6.44]; on 2009 FLORES lines 6.10× [6.05, 6.15] | the sample can't distinguish "5.4×" from "6.4×" |
| `seed` | `random` used only on lines 21 and 25; three seeds → identical output | **harmless** (dead code) |
| `special` | XLM-R with special tokens: +2 tokens per line (+6.7% eng, +5.4% hin) | v0's `add_special_tokens=False` is **correct** |
| `crash` | v0 on a line containing `<|endoftext|>` → exit 1 (tiktoken disallowed special) | robustness bug, doesn't affect reported numbers |

**NFC — the result I didn't expect.** Re-tested on RAW tarball text (fixing Entry 3's invalid test):
- On the starter-kit sample: NFC changes nothing (0.0% in Entry 3 — the sample has no affected code points).
- On real FLORES: NFC *decomposes* precomposed nukta letters (U+09DF BENGALI YYA in 1139 lines, Devanagari ZA/DDDHA/FA…). These are Unicode "composition exclusions", so NFC turns one code point into two.
- Token effect depends on the tokenizer: gpt2 Hindi +0.12%; Llama Bengali +1.45%; **sarvam-1 Bengali −12.76%**, Hindi −1.07%.

My first instinct was "NFC changes 1195 Bengali lines → it's a bug". **Wrong framing.** NFC is the standard canonical form; it never changes meaning, and on the data the report used it has exactly zero effect. Verdict: **not a bug in v0**. But it is a real benchmark caveat: the eval should normalise the same way the production request path does, otherwise Bengali numbers can be off by up to ~13% for some tokenizers. Added to A1/A3 caveats.

---

## Entry 6 — Is the premium stable across domains?

`python partA/fertility_v1.py --tokenizer hf:unsloth/Llama-3.2-1B --tokenizer hf:sarvamai/sarvam-1 --langs eng,hin,kan,tam,tel,mal --by-domain`

Llama-3.2 Hindi: wikibooks 2.53×, wikinews 2.51×, wikivoyage 2.54×. Dravidian varies ±0.2× across domains.
**So what:** stable *within* formal Wikipedia-style text. Says nothing about chat, code-mixed or romanised input — which is what production traffic looks like. That is the biggest caveat for A4.

---

## Entry 7 — Part B from the spec, then checked against the log

**Command:** `python partB/capacity.py` (all arithmetic in the script; spec values parsed from `model_spec.md`).

**B1(a):** 2 × 28 × 8 × 128 × 2 B = **114,688 B/token** (112 KiB). Easy to get 3× wrong by using the 24 Q heads; GQA means only the 8 KV heads are cached.

**B1(b) — first dead end:** I computed the pool with "24 GB" = 24 GiB and got 28.9 sequences. Then re-read the spec: it says "GB". Decimal reading gives **25.7 sequences** (105,329 tokens). Rather than argue units, I let the log decide: implied capacity = resident tokens / kv_cache_util is 102k–107k tokens in every no-preemption row with util ≥ 0.1 (the 0.01–0.06 rows are too rounded to use). That matches 105k (decimal), not 118k (GiB). Prediction confirmed: batch 24 × 4096 = 98,304 tokens → util 0.93 predicted 0.933; batch 32 × 4096 = 131,072 > 105k → first preemptions (7) exactly where predicted.

**B3 — the misread column:** hypothesis "reported_tok_s includes prompt tokens". Test: recompute n × (prompt+gen) / wall for all 13 rows → matches every row to <0.5%. Gen-only does not match any row. So the harness counter counts prefill tokens as "throughput". Consequences:
- batch 16: report says long 1311 > short 883. Goodput: short 294.5 vs long 163.9 gen tok/s → short prompts produce **1.80× more** output per second. The report's conclusion is reversed.
- "batch 48 → ~3200 tok/s": the log *already contains* a batch-48 row: 1298.5 reported, 162.3 goodput. The extrapolation ignored a measured row.

Goodput batch 24 long, two independent ways: counts/wall = 200.9 tok/s; latency columns 24/ITL = 249.8 tok/s. **They disagree by 24%**, which bothered me. Resolution: TTFT + 511×ITL = 49.6 s per request but wall is 61.2 s → 11.6 s of the run is not steady decode (prefill of 86k prompt tokens and ramp-up). 200.9 is the end-to-end number a capacity plan should use; 249.8 is the decode-phase ceiling.

**B2 — mechanism:** long sweep goodput 70.7 → 112.8 → 163.9 → 200.9 (bs 4→24), then **173.0 (bs 32), 162.3 (bs 48)**. In the same rows: preempted_seqs 0 → 7 → 23, kv_cache_util 0.93 → 0.97, TTFT 500 → 637 → 955 ms. Key observation: **ITL is flat** (96.1 → 101.8 → 100.0 ms) while the non-decode part of wall clock (wall − 512×ITL) jumps 12.0 → 42.6 → 100.2 s. So individual decode steps aren't getting slower; the GPU runs at most ~25 sequences at a time (KV-bound) and the extra time is preempted sequences waiting and re-prefilling.

**Fix — surprise:** I expected `max_num_seqs=24` to help both overloaded rows. Predicting from the log's own rows (waves of 24, using the measured bs-24 and bs-8 wall clocks): bs 48 → 122.3 s vs 151.4 s measured (**+24% goodput**), but bs 32 → 97.5 s vs 94.7 s (**−3%**) because the second wave of 8 is small and inefficient. So the cap is only a win under sustained overload. Kept it as the proposal (it's zero-risk to quality and the bs-48 case is the one leadership asked about), and added fp8 KV cache as the alternative: halves bytes/token → 51.4 seqs fit; rough model (ITL fit + measured prefill rate) predicts ~319 gen tok/s at bs 48, ~2× today — rougher, depends on fp8 read cost assumption and quality.

**Data-quality catch:** e2e_ms_p95 > wall_clock_s in 12/13 rows (bs 1: 12.86 s p95 for a 10.94 s run). Impossible on the same clock with simultaneous submission. I don't know which clock e2e uses, so I don't use that column for anything.

---

## Entry 8 — Reproducibility run + checking the write-ups against the outputs

**Command:** `PY=../.venv/Scripts/python bash run_all.sh` — reruns every experiment, rewrites `partA/results/*` and `partB/capacity_output.txt`.

**Dead ends:** first two attempts died immediately — (1) relative interpreter path broke after `cd starter_kit`; (2) the project path contains a space, so `$PY` had to be quoted. Third run: exit 0, every number identical to Entries 1–7 (all scripts are deterministic; bootstrap uses fixed seeds).

**Errors I caught in my own drafts by re-checking against the result files:**
- A2 #2: I first wrote "at most 4 sample sentences have a counterpart". Recounting: 5 (book, cricket, train, cupboard, Mysuru). Fixed.
- A2 #1: "overstates Dravidian by 30–46%" → actual 29% (Telugu) to 46% (Malayalam). Fixed.
- A1: I had written that romanised Hindi "would be close to 1×" — not measured, so removed (evidence rule applies to my caveats too).
- A4: claimed the real-tokenizer rerun "takes about a minute" — not measured (the 4-tokenizer run with bootstrap takes several minutes). Removed. Also "cuts tokens 5–6×" → recomputed 4.5× (Bengali) to 6.2× (Kannada).
- Part C: "+15 s" latency → 1500 tok × ~11 ms = 16.5 s; reviewer coverage of synthetic data "~5%" → 1800/12000 = 15% upper bound. Fixed.

**Lesson:** every number in prose drifts from the script output unless re-checked. Final pass = grep each number in the write-ups against `partA/results/` and `partB/capacity_output.txt`.
- Part C memo was 796 words (over one page) and had a markdown table row broken across two lines. Rewrote to 628 words; content and numbers unchanged.

---

## Entry 9 — My own re-run to verify everything (Windows, Python 3.12)

**What I did:** Installed the requirements fresh (`pip install -r requirements.txt`), then ran every script one by one to confirm they work on my machine and that the numbers match what is written in the write-ups.

**Commands run:**
```
python fertility.py --corpus eng=corpus_sample/eng_sample.txt --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
python partA/get_corpus.py
python partA/audit_ablation.py --pair sample
python partA/audit_ablation.py --pair flores
python partA/audit_extras.py spaces
python partA/audit_extras.py lower
python partA/audit_extras.py bootstrap
python partA/audit_extras.py seed
python partA/audit_extras.py special
python partA/audit_extras.py crash
python partA/audit_extras.py nfc
python partA/fertility_v1.py --tokenizer gpt2 --tokenizer hf:unsloth/Llama-3.2-1B --tokenizer hf:sarvamai/sarvam-1 --tokenizer hf:FacebookAI/xlm-roberta-base --langs eng,hin,kan,tam,tel,mal,ben,mar --by-domain --out partA/results/main
python partB/capacity.py
```

**Result:** All numbers matched exactly.

| check | expected | got |
|---|---|---|
| baseline ratio | 5.89× | 5.89× ✓ |
| FLORES ablation, no lowercasing | +3.6% | +3.6% ✓ |
| KV bytes/token | 114,688 B | 114,688 B ✓ |
| KV pool capacity | 25.7 seqs | 25.7 seqs ✓ |
| goodput batch-24 long | 200.9 gen tok/s | 200.9 gen tok/s ✓ |
| Hindi premium gpt2 | 7.44× | 7.44× ✓ |
| Hindi premium sarvam-1 | 1.14× | 1.14× ✓ |

**Things I also checked manually:**
- Re-derived B1 by hand: `2 × 28 × 8 × 128 × 2 = 114,688`. Pool: `(24 × 10⁹ × 0.92 − 8.4 × 10⁹ − 1.6 × 10⁹) ÷ 114,688 = 25.7 seqs`. Matches.
- Confirmed B3 goodput reversal: short-prompt batch-16 goodput `294.5 gen tok/s` vs long `163.9` — the report's "long is better" uses inflated reported_tok/s that counts prompt tokens.
- Read the Hindi translations in A2 #2. The sentence pairs that roughly match (book, cricket, train, cupboard, Mysuru) are translated correctly.

**One issue on Windows:** `audit_extras.py sample` threw a `UnicodeEncodeError` when printing Hindi to the Windows terminal (cp1252 can't render Devanagari). The numbers before the crash are correct; this does not affect any submitted results file.

**Conclusion:** every result file in `partA/results/` and `partB/capacity_output.txt` is reproducible from scratch on my machine with the exact commands in `README.md`.
