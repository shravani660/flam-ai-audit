# AI usage

## What I used
**Claude Code (Claude Opus 5.5)**, in one long session. It read the assignment, wrote the plan, wrote every script in
`partA/` and `partB/`, ran all the experiments locally, and drafted every write-up. I directed the work, and afterwards
re-ran and checked it (section 3).

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
<!-- FILL THIS IN YOURSELF, honestly, after you have done it. Examples of what to write: -->
<!-- - Re-ran `bash run_all.sh` on my machine; numbers matched / differed in ... -->
<!-- - Re-derived B1 (114,688 B; 25.7 seqs) by hand on paper. -->
<!-- - Checked the Hindi translations in A2 #2. -->
<!-- - Parts I still don't fully understand: ... -->
- TODO
