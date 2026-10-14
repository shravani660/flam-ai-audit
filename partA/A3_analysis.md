# A3 — Corrected cross-language analysis

**Command** (output in `partA/results/04_main.txt` and `partA/results/main.csv`):
```
python partA/fertility_v1.py --tokenizer gpt2 --tokenizer hf:unsloth/Llama-3.2-1B \
  --tokenizer hf:sarvamai/sarvam-1 --tokenizer hf:FacebookAI/xlm-roberta-base \
  --langs eng,hin,kan,tam,tel,mal,ben,mar --by-domain --out partA/results/main
```
Corpus: FLORES-200 dev+devtest, 2009 parallel sentences (A1). Fixes from A2 are applied: no lowercasing, `split()`,
ratio of totals. NFC is applied.

**Tokenizers (4).**
- **gpt2**: what v0 used.
- **Llama-3.2**: 128,256 vocab, a *proxy* for our 128k-vocab FLM-4B, whose tokenizer is not in the kit.
- **sarvam-1**: Indic-focused LLM tokenizer, 68k vocab.
- **XLM-R**: multilingual, 250k vocab.

The [UNK] rate is 0.00% for all four, so no tokenizer looks cheap by discarding text.

**Denominators (4).** Per parallel sentence, per whitespace word, per grapheme cluster, per UTF-8 byte.

## Headline: premium vs English = tokens(lang) / tokens(eng) on the same 2009 sentences

| lang | gpt2 | **Llama-3.2 (our-model proxy)** | XLM-R | sarvam-1 |
|---|---|---|---|---|
| hin | 7.44× | **2.53×** [2.51, 2.55] | 1.25× | 1.14× |
| mar | 7.83× | **2.79×** | 1.22× | 1.07× |
| ben | 9.63× | **5.86×** | 1.38× | 1.11× |
| tam | 15.48× | **7.63×** | 1.35× | 1.16× |
| tel | 13.00× | **8.32×** | 1.33× | 1.16× |
| kan | 13.59× | **8.87×** | 1.36× | 1.23× |
| mal | 15.15× | **8.94×** | 1.37× | 1.35× |

Brackets show the paired bootstrap 95% CI (2000 resamples of sentence ids). Every CI is within about ±1% of its point
estimate, so with 2009 sentences sampling noise is no longer the issue. Domain and representativeness are (see A1).

## Same tokens, four denominators (Llama-3.2 proxy)

| lang | tok / sentence | tok / word | tok / grapheme | tok / byte | ratio vs eng **per word** | ratio vs eng **per sentence** |
|---|---|---|---|---|---|---|
| eng | 26.4 | 1.24 | 0.206 | 0.206 | 1.00× | 1.00× |
| hin | 66.6 | 2.66 | 0.795 | 0.203 | 2.16× | 2.53× |
| kan | 233.8 | 14.90 | 2.660 | 0.641 | 12.05× | 8.87× |
| tam | 201.2 | 12.28 | 2.081 | 0.493 | 9.94× | 7.63× |
| tel | 219.4 | 13.22 | 2.935 | 0.636 | 10.70× | 8.32× |
| mal | 235.7 | 16.15 | 3.053 | 0.590 | 13.07× | 8.94× |

Read across a row: Hindi tok/byte (0.203) is almost the same as English (0.206). That doesn't mean Hindi is cheap.
Devanagari simply needs about 2.6× more bytes than English to say the same sentence (328.6 vs 128.1 bytes per sentence).
Tok/word, meanwhile, overstates Kannada (12.05× vs 8.87×) and understates Hindi (2.16× vs 2.53×). Every
writing-system-based denominator answers a different question.

## Which single number should drive routing and cost?

**Tokens per parallel sentence, measured on the tokenizer of the model that will actually serve the traffic.**
Expressed for a decision, that is the *premium vs English* above, or the absolute tokens per sentence when comparing two
models.

**Why.**
1. We pay for, and wait on, **tokens**. Cost and latency scale with the numerator.
2. The denominator has to hold constant **what the user wants to say**, because that is the thing that is the same
   across languages. A parallel sentence holds content constant by construction.
3. Words, characters, graphemes and bytes are properties of each writing system, not of content. They change across
   languages for reasons unrelated to cost: agglutination, separate postpositions, 3-byte UTF-8 encoding of Indic
   scripts. A ratio per word or per byte is therefore partly a ratio of writing systems. A2 #1 and #6 show these errors
   go in opposite directions for different languages.
4. It must be measured on *our* tokenizer, because A2 #7 shows the tokenizer, not the script, decides the answer. The
   same Hindi sentence costs 7.44× on gpt2 and 1.14× on sarvam-1.

**One subtlety for routing *between* models.** "Premium vs English" is relative to each tokenizer's own English cost,
and English cost differs (26.4 tokens per sentence on Llama-3.2, 30.5 on sarvam-1). So to compare two candidate models,
use **absolute tokens per parallel sentence × that model's cost per token**:

| content | Llama-3.2 proxy tok/sent | sarvam-1 tok/sent | sarvam / proxy |
|---|---|---|---|
| English | 26.4 | 30.5 | 1.16× (sarvam costs more) |
| Hindi | 66.6 | 34.7 | 0.52× |
| Kannada | 233.8 | 37.5 | 0.16× |
| Malayalam | 235.7 | 41.3 | 0.18× |

## What changes vs REPORT_v0
- "Hindi costs ~6×" was measured on the wrong tokenizer (gpt2), with the wrong denominator (words), on 10 non-parallel
  sentences. On our-model proxy, Hindi is about **2.5×**, not 6×.
- The report never measured the Dravidian languages. On the proxy they are the expensive ones, at **7.6–8.9×**, with
  Bengali at 5.9×. That is where routing would save money.
- "Property of the script" is false. An Indic-aware vocabulary brings every language here to 1.07–1.35× of English.

**Limits of this analysis.** It covers input text only, formal register only, and uses a proxy tokenizer. Token count
is cost, not quality: a model with an efficient Indic tokenizer is only a routing target if its answers are as good.
