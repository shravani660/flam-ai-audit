# A2 — Audit of `fertility.py` and the metric

**How to read this.** Every finding below has a claim, the exact command, the before/after numbers, and why the delta
proves the claim. All commands are run from `submission/`, and outputs are saved in `partA/results/`.

**Control.** `audit_ablation.py` imports the original `analyze()` and asserts that its own default configuration gives
identical numbers ("control OK"). So every delta below is caused only by the one switch that was flipped.

**Baseline** (`results/00_baseline_v0.txt`): eng 1.27 tok/word, hin 7.45 tok/word, ratio **5.89×**. This reproduces
REPORT_v0 exactly.

---

## Summary

| # | finding | type | effect on hin/eng ratio | direction |
|---|---|---|---|---|
| 1 | Tokens-per-word is the wrong cross-language quantity | **conceptual** | gpt2 Hindi 6.34× per word vs **7.44× per parallel sentence**; Malayalam 22.15× vs **15.15×** | understates Hindi, overstates Dravidian |
| 2 | Sample corpora are not parallel and too small | data | 5.89×, but 95% CI [5.37, 6.44] | ±9%, any direction |
| 3 | `.lower()` changes only the English side | code bug | 5.89 → 6.06 (+2.9%); FLORES +3.6% | understates the gap |
| 4 | `split(" ")` counts empty strings as words | code bug | +0.6% (sample); word count +3.4% for Kannada | understates fertility |
| 5 | Mean of per-line ratios instead of ratio of totals | code bug (aggregation) | +0.3% / +0.2% | small |
| 6 | tok/char "confirming" tok/word is not independent evidence | report logic | per-char ratio is 2.65×, 7.00× or 10.86× depending on what "char" means | — |
| 7 | "Property of the script, not the tokenizer" | report logic | same Hindi text: gpt2 7.44× vs sarvam-1 1.14× | refuted |
| 8 | gpt2 is not our serving tokenizer | report scope | gpt2 vocab 50,257 vs our model's 128k | — |
| 9 | Crash on text containing `<|endoftext|>` | robustness bug | doesn't change the reported numbers | — |
| H1 | `random.seed(1337)` | **harmless** | 0.0% | — |
| H2 | NFC normalisation | **harmless for v0** (measured), caveat for production | 0.0% on the report's data | — |
| H3 | `add_special_tokens=False` | **correct** | would add +5–7% if changed | — |

---

## 1. Conceptual: tokens-per-word measures the wrong thing

**Claim.** The code correctly computes tokens ÷ whitespace words. But a "word" is not the same amount of meaning in every
language, so tokens/word cannot compare cost across languages. What you want held constant is the *content*. Only a
parallel sentence holds content constant.

**Command.** `python partA/fertility_v1.py --tokenizer gpt2 --langs eng,hin,kan,tam,tel,mal`

**Numbers (FLORES, 2009 parallel sentences, `results/04_main.txt`):**

| lang | words / sentence | gpt2 ratio per word (v0's metric) | gpt2 ratio per sentence (same content) |
|---|---|---|---|
| eng | 21.3 | 1.00× | 1.00× |
| hin | 25.0 | 6.34× | **7.44×** |
| kan | 15.7 | 18.47× | **13.59×** |
| tam | 16.4 | 20.16× | **15.48×** |
| tel | 16.6 | 16.72× | **13.00×** |
| mal | 14.6 | 22.15× | **15.15×** |

**Why this proves it.** The tokens are identical in both columns; only the denominator changes. Hindi writes postpositions
and auxiliaries as separate words (में, को, है), so it has *more* words than English for the same sentence. Per-word
therefore *understates* Hindi's real cost by 15% (6.34 vs 7.44). Dravidian languages are agglutinative: one long word
carries what English says in several. Per-word therefore *overstates* them by 29–46% (Telugu 16.72/13.00 to Malayalam 22.15/15.15). A metric that errs in opposite
directions for different languages cannot rank them.

## 2. The sample corpora are not parallel, and 10 lines is too few

**Claim.** REPORT_v0 calls the files "parallel line-by-line". They are not. They are two unrelated sets of 10 sentences.

**Command.** `python partA/audit_extras.py sample` prints them side by side (`results/03_extra_sample.txt`).
Translations of the Hindi lines are mine. They are simple, and any Hindi reader can verify them.

| line | English file | Hindi file (translation) |
|---|---|---|
| 1 | Bengaluru International Airport handled record traffic in March. | I like morning tea a lot. |
| 2 | The Quarterly Review meeting moved to Thursday. | It is raining lightly in Bengaluru today. |
| 3 | I bought this book yesterday from a small shop near MG Road. | I bought this book just yesterday. |
| 4 | Children are playing cricket on the ground. | We are going to Mysore next week. |
| 6 | NASA and ISRO announced a joint mission update. | The train arrived right on time. |

5 of the 10 English sentences have a rough counterpart somewhere in the Hindi file (book, cricket, train, cupboard,
Mysuru). Only one of them sits on the same line (line 3), and that one is a shorter version. The
English side contains acronyms and names (NASA, ISRO, GPU, MG Road) that the Hindi side does not.

**Measured consequence (sample size).** `python partA/audit_extras.py bootstrap` resamples lines 5000 times:

| data | v0 ratio | 95% bootstrap CI |
|---|---|---|
| sample, 10 lines | 5.89× | [5.37, 6.44] |
| FLORES, 2009 lines | 6.10× | [6.05, 6.15] |

**Measured consequence (different content).** If you pretend the sample is parallel and compute tokens per sentence,
`python partA/fertility_v1.py --tokenizer gpt2 --corpus-dir starter_kit/corpus_sample --file-pattern "{lang}_sample.txt" --langs eng,hin`
gives a Hindi premium of **4.78×**. On truly parallel FLORES it is **7.44×**. The sample's Hindi sentences are shorter
and simpler (6.1 words vs 7.8 for English), so a same-content comparison is impossible with these files.

**Why this proves it.** With 10 lines, the "5.89×" cannot be told apart from 5.4× or 6.4×. The interval is ten times
narrower with 2009 lines. And because the content differs, even an exact number would not be a same-content comparison.

## 3. `.lower()` changes only one side

**Claim.** Lowercasing changes English and leaves Hindi alone (Devanagari has no case). This is asymmetric preprocessing,
and it also measures text users don't actually send.

**Command.** `python partA/audit_ablation.py --pair sample` and `--pair flores`, row "no lowercasing".

| | eng t/w | hin t/w | ratio |
|---|---|---|---|
| sample, with lower (v0) | 1.265 | 7.448 | 5.89× |
| sample, without lower | 1.229 | 7.448 | 6.06× (**+2.9%**) |
| FLORES, with → without | 1.285 → 1.241 | 7.844 → 7.844 | 6.10× → 6.32× (**+3.6%**) |

Mechanism (`python partA/audit_extras.py lower`): gpt2 learned capitalised words as single tokens. "European",
"September" and "Olympic" go from 1 to 3 tokens when lowercased. On FLORES, lowercasing adds tokens to 1629 English
word occurrences and removes them from only 214. In the sample it hits 3 lines (NASA/ISRO, GPU, Quarterly Review).

**Why this proves it.** The Hindi column doesn't move at all and the English column does, so the ratio shift is caused
entirely by the one-sided transformation. The direction is the opposite of the usual intuition: lowercasing makes English
*more* expensive, so v0 *understates* the gap by about 3–4%.

## 4. `split(" ")` counts empty strings as words

**Claim.** `"a  b".split(" ")` returns `['a', '', 'b']`, i.e. 3 "words". Every run of extra spaces adds phantom words, which
pushes fertility down.

**Command.** `python partA/audit_extras.py spaces` and the ablation row "split() instead of split(' ')".

| corpus | phantom words | word-count inflation |
|---|---|---|
| sample eng / hin | 1 / 1 (the double spaces on eng line 7, hin line 10) | ratio 5.89 → 5.92 (+0.6%) |
| FLORES eng | 1 | +0.00% |
| FLORES hin | 15 | +0.03% |
| FLORES tel | 449 | +1.35% |
| FLORES kan | 1069 | **+3.39%** |

**Why this proves it.** `split()` and `split(" ")` differ only in how they handle whitespace runs, so the whole delta is
phantom words. It is negligible for eng/hin but material for Kannada, which is one of the languages we must compare.

## 5. Mean of per-line ratios instead of ratio of totals

**Claim.** `analyze()` averages each line's tokens/words. That gives a 3-word line the same weight as a 40-word line.
Cost is total tokens, so the right aggregate is Σtokens / Σdenominator.

**Command.** Ablation row "ratio of totals". Ratio 5.89 → 5.91 (sample, +0.3%); 6.10 → 6.12 (FLORES, +0.2%).

**Why this proves it, and the honest magnitude.** This is a real correctness issue, but on these corpora it moves the
headline by only 0.2–0.3%. I don't treat it as a headline finding.

## 6. tok/char "agreeing" with tok/word is not independent confirmation

**Claim.** REPORT_v0 finding 2 says the 7.0× per-character number "confirms" the 5.89× per-word number, so the result
is "robust". Both metrics share the same numerator (tokens), and they don't even agree: 5.89 vs 7.00 is a 19% gap.
The per-char number also depends entirely on what "character" means.

**Command.** Ablation rows "graphemes" and "UTF-8 bytes" (tok/char column).

| "character" = | hin/eng tok-per-char ratio (sample) |
|---|---|
| Unicode code point (v0's `len(line)`) | 7.00× |
| grapheme cluster (what a reader sees as one letter) | 10.86× |
| UTF-8 byte | 2.65× |

**Why this proves it.** Same tokens, three answers spanning 4×. The per-char ratio is mostly a statement about how
Devanagari is encoded: about 1.5 code points and about 3.9 bytes per visible letter. It isn't evidence about cost.

## 7. "Property of the script, not the tokenizer" is false

**Claim.** If Hindi's script made *any* tokenizer struggle, an Indic-aware tokenizer would show a similar gap.

**Command.** `python partA/fertility_v1.py --tokenizer gpt2 --tokenizer hf:unsloth/Llama-3.2-1B --tokenizer hf:sarvamai/sarvam-1 --tokenizer hf:FacebookAI/xlm-roberta-base`

| tokenizer | Hindi premium vs English (tokens per parallel sentence) |
|---|---|
| gpt2 | 7.44× |
| Llama-3.2 (128k) | 2.53× |
| XLM-R | 1.25× |
| sarvam-1 | 1.14× |

**Why this proves it.** The text and the script are identical across rows, yet the premium varies 6.5×. The cause is
the tokenizer's vocabulary. On gpt2, the Dravidian languages tokenize at 0.98–1.00 tokens per UTF-8 byte, i.e. raw byte
fallback with essentially no learned merges.

## 8. The report benchmarked a tokenizer we don't serve

**Claim.** `bench/model_spec.md` says our model has a 128k vocabulary. The report ran `gpt2` (50,257 tokens; vocab
size printed by `fertility_v1.py`). So the report's cost numbers are for a tokenizer that isn't in our stack. The FLM-4B
tokenizer isn't in the kit, so I use Llama-3.2 (128,256) as a labelled proxy. Hindi is 7.44× on gpt2 and 2.53× on the
proxy (from #7). **Caveat:** the proxy is my assumption. The real tokenizer must be run before any budget is set.

## 9. Crash on `<|endoftext|>`

**Command.** `python partA/audit_extras.py crash` → v0 exits with code 1 (tiktoken refuses disallowed special tokens).
Any corpus containing that string, such as scraped chat logs, kills the run. It doesn't affect the reported numbers. v1
passes `disallowed_special=()`.

---

## Things that look suspicious but are fine

**H1 — `random.seed(1337)`.** `random` appears only on lines 21 (import) and 25 (seed), and nothing ever samples.
`python partA/audit_extras.py seed` runs the script with seeds 1337, 0 and 42 and gets **1 distinct output**. It is dead
code with no effect.

**H2 — NFC normalisation (`unicodedata.normalize("NFC", …)`).** On the report's data it has exactly **0.0% effect**
(ablation row "no NFC", sample). NFC is the standard canonical form and never changes meaning. *What I got wrong first:*
on raw FLORES, NFC changes 1195 Bengali lines, and my first reaction was "bug". Measuring showed the token effect is
tokenizer-dependent (gpt2 Hindi +0.12%, sarvam-1 Bengali −12.8%). That isn't a bug in v0. It is a benchmark-design
caveat: normalise the same way the serving path does. I carry it into A1 and A4.

**H3 — `add_special_tokens=False`.** It looks like it might be undercounting. `python partA/audit_extras.py special`
shows that turning it on adds exactly 2 tokens per line (`<s>`, `</s>`): +6.7% English and +5.4% Hindi for XLM-R. Those
are not text tokens, and adding them would distort short lines most. v0 is correct.

**Also fine:** `strip()` and skipping empty lines don't change any number on the sample files, which have no blank lines
and no leading or trailing spaces. Note, though, that skipping
blank lines would silently break alignment on a parallel corpus with a blank line in only one language. v1 keeps
placeholders and drops that sentence id in all languages.

## Corrected version
`partA/fertility_v1.py` fixes #3–#5 and #9, adds the per-sentence, per-grapheme and per-byte denominators, a paired
bootstrap CI, and an [UNK] rate. It refuses input that isn't parallel. Results are in A3.
