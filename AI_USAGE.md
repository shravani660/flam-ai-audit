# AI usage

## What I used
**Claude Code (Claude Sonnet / Opus)**, across multiple sessions. I used it to read the assignment, plan the approach,
write the scripts in `partA/` and `partB/`, run the experiments, and draft the write-ups. I directed every step,
reviewed each output, and afterwards re-ran everything and verified the numbers myself (section 3).

## Where it helped
- Turning a vague brief into a concrete plan (`PLAN.md`) and a reproducible pipeline (`run_all.sh`).
- The ablation design: re-implementing `analyze()` behind switches, with a control that asserts it matches the
  original exactly. That makes every "this is a bug" claim isolated and measurable.
- Finding FLORES-200's direct download (the Hugging Face mirrors are gated).
- Part B arithmetic, and spotting that `reported_tok_s` = (prompt + gen) × n / wall in all 13 rows.

## Where it misled me, or was wrong before measuring
All of these are visible in NOTEBOOK.md:
1. **Lowercasing direction.** Its hypothesis was "lowercasing makes English cheaper". The measurement showed the
   opposite: gpt2 learned capitalised words as single tokens, so lowercasing *adds* English tokens (Entry 3).
2. **NFC.** Its first reaction to "NFC changes 1195 Bengali lines" was "bug". Measuring showed zero effect on the
   report's data. It is a production caveat, not a v0 bug (Entry 5).
3. **GB vs GiB.** Its first B1 calculation used GiB (28.9 sequences). The log's `kv_cache_util` showed decimal GB
   (25.7) is right (Entry 7).
4. **`max_num_seqs=24`.** It expected the fix to help batch 32 too. The prediction from the log's own rows says −3%;
   it only helps batch 48 (+24%) (Entry 7).
5. **Numbers drifting in prose.** It miscounted parallel sample lines (4 vs 5), rounded 29% to 30%, and wrote
   unmeasured claims ("about a minute", "romanised Hindi ≈ 1×"). Each was caught by re-checking against the result
   files and fixed (Entry 8).
6. **The translations** of the 10 Hindi sample sentences in A2 #2 are the AI's. They have to be checked by a Hindi
   reader (section 3).

## What I verified myself
- Re-ran all scripts individually from the `submission/` folder on my machine (Windows, Python 3.12). Every number
  in `partA/results/` and `partB/capacity_output.txt` came out identical to what is already committed:
  baseline `eng 1.27 / hin 7.45 / 5.89×`; FLORES ablation `6.10→6.32×` without lowercasing; capacity
  `114,688 B/token`, `25.7 sequences`, goodput `200.9 gen tok/s` at batch 24.
- Re-derived B1(a) by hand: `2 × 28 × 8 × 128 × 2 = 114,688 bytes`. Then B1(b): `(24×10⁹ × 0.92 − 8.4×10⁹ − 1.6×10⁹) ÷ 114,688 = 105,329 tokens ÷ 4096 = 25.7 sequences`. Matches the script.
- Checked the B3 goodput reversal myself: batch-16 short prompts `294.5 gen tok/s` vs long `163.9 gen tok/s` —
  the report's "long is better" conclusion is based on reported throughput that counts prompt tokens, not output tokens.
- Read through the Hindi translations in A2 #2. The 5 matched pairs (book, cricket, train, cupboard, Mysuru)
  look correct in meaning; sentences are simpler Hindi than the English side, which is exactly what the write-up says.
- Looked at the NFC result: on the report's own data (the sample) the effect is 0.0%, so it is not a bug in v0.
  On raw FLORES it matters for Bengali with sarvam-1 (−12.76%), which I would flag as a caveat before any Bengali deployment.
- Parts I am less confident about: the ITL linear-fit slope used in the fp8 prediction (Entry 7); I confirmed the
  formula but haven't independently measured the GPU bandwidth assumption.
