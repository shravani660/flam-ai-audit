# A4 — Memo: tokenizer cost premium for Indic traffic

**To:** leadership (routing & capacity) **Re:** REPORT_v0 Section 1 **Status:** replaces the "6× Hindi" number

## Corrected headline numbers
Measured as tokens needed to say *the same sentence* vs English, on FLORES-200 (2009 parallel sentences). The Llama-3.2
128k tokenizer stands in for our 128k-vocab model.

| | Hindi | Marathi | Bengali | Tamil | Telugu | Kannada | Malayalam |
|---|---|---|---|---|---|---|---|
| our-model proxy (Llama-3.2) | **2.5×** | 2.8× | 5.9× | 7.6× | 8.3× | 8.9× | 8.9× |
| Indic-aware tokenizer (sarvam-1) | 1.1× | 1.1× | 1.1× | 1.2× | 1.2× | 1.2× | 1.4× |

95% confidence intervals are within ±1%. REPORT_v0's "Hindi 5.89×" came from the wrong tokenizer (gpt2, not ours), the
wrong metric (per word, which is biased in opposite directions for Hindi and for Dravidian languages), and 10 unrelated
sentences (CI 5.4–6.4×). It never measured the South Indian languages, which are where the real cost sits.

## Recommendation
1. **Don't budget "6× for Hindi".** On our-model proxy Hindi is about 2.5×. Budget per language using the table above.
2. **Before any routing decision**, run our actual FLM-4B tokenizer through `partA/fertility_v1.py`. It takes one
   command (`--tokenizer hf:<our-repo>`). All numbers here assume the proxy behaves like our tokenizer.
3. **If our tokenizer matches the proxy**, routing pays off first for **Tamil, Telugu, Kannada, Malayalam and Bengali**
   (6–9×). An Indic-efficient tokenizer would cut their tokens by 4.5–6× (Bengali 155 → 34, Kannada 234 → 38 tokens per
   sentence).
   Hindi and Marathi (about 2.5–2.8×) are a lower priority.
4. **Routing is conditional on quality.** Fewer tokens is only a saving if the Indic model's answers are as good. Run
   a quality evaluation on the candidate model before switching traffic. Note that an Indic tokenizer costs about 16%
   *more* on English (30.5 vs 26.4 tokens per sentence), so route by detected language, not wholesale.

## Biggest caveat
FLORES is formal, translated, Wikipedia-style text. Real user messages are short, casual, often **code-mixed or
romanised** ("kal meeting hai"). Neither is measured here, and the premium could be very different on them. These
numbers also cover only *input* text. Output length per language, which drives decode cost, is not measured.

## The one production metric to watch
**Mean billed tokens per request (prompt + completion) for each detected language, divided by the same number for
English**, computed daily from serving logs with our real tokenizer.

If this analysis is right, it should land near the table above: Hindi about 2.5×, Dravidian about 8×. If the live ratio
for any language is off by more than ±25% for a week, the FLORES numbers don't transfer to our traffic, and the routing
case has to be re-made on a sample of real requests.

This metric assumes users ask similar things in every language. That is the obvious confound, and a large divergence is
exactly the signal to go and look.
