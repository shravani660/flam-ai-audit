# A1 — Evaluation corpus

## What I used
**FLORES-200**, dev + devtest splits (Meta / NLLB team), downloaded from the original tarball
`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`. Licence CC-BY-SA 4.0.

**Why this corpus:** the question is "how many tokens does it cost to say *the same thing* in language X vs English".
That needs a **parallel** corpus: the same sentences, professionally translated. FLORES gives exactly that for every language I need,
and it is the standard multilingual benchmark, so others can reproduce my numbers.

**Reproduce:** `python partA/get_corpus.py` → writes `partA/corpus/<lang>.txt` and `partA/corpus/domains.tsv`, then asserts every file has the same line count.

## Size (output of `partA/get_corpus.py` and `partA/fertility_v1.py`)

| lang | FLORES code | family | sentences | words (`split()`) | words / sentence | graphemes / sentence | UTF-8 bytes / sentence |
|---|---|---|---|---|---|---|---|
| eng | eng_Latn | Indo-European | 2009 | 42,855 | 21.3 | 128.0 | 128.1 |
| hin | hin_Deva | Indo-Aryan | 2009 | 50,250 | 25.0 | 83.8 | 328.6 |
| kan | kan_Knda | **Dravidian** | 2009 | 31,530 | 15.7 | 87.9 | 364.7 |
| tam | tam_Taml | **Dravidian** | 2009 | 32,909 | 16.4 | 96.7 | 408.4 |
| tel | tel_Telu | **Dravidian** | 2009 | 33,326 | 16.6 | 74.7 | 344.7 |
| mal | mal_Mlym | **Dravidian** | 2009 | 29,307 | 14.6 | 77.2 | 399.7 |
| ben | ben_Beng | Indo-Aryan | 2009 | 38,262 | 19.0 | 79.6 | 339.6 |
| mar | mar_Deva | Indo-Aryan | 2009 | 37,111 | 18.5 | 77.8 | 344.0 |

Required minimum was 4 languages (eng, hin, 2 Dravidian). I used 4 Dravidian languages. I also added Bengali and Marathi
because the Part C product scenario targets them; they cost nothing extra to include.

**Domain:** Wikipedia-family text in three domains: wikinews 689, wikivoyage 668, wikibooks 652 sentences (`domains.tsv`).
Average English sentence is 21 words, which is long and formal.

## Preprocessing
- Strip leading/trailing whitespace per line. Nothing else is removed: no lowercasing, no space collapsing, no filtering of
  numbers, Latin-script names or punctuation. We want to measure the text users actually send.
- **NFC normalisation.** This is a deliberate choice with a measured side effect. NFC changed 183/2009 Hindi lines and
  1195/2009 Bengali lines, because it decomposes precomposed nukta letters such as U+09DF BENGALI YYA. The effect on token
  counts depends on the tokenizer: gpt2 Hindi +0.12%, Llama-3.2 Bengali +1.45%, sarvam-1 Bengali **−12.8%**
  (`audit_extras.py nfc`). The production path should normalise the same way the benchmark does.
- Alignment is preserved by construction: the same line index in every file is the same sentence. `get_corpus.py` exits
  if the line counts differ.

## What this corpus cannot tell us
FLORES is formal, edited, encyclopedic text translated *from English*. Our users send short, casual chat messages. Real
Indic chat is often **code-mixed** ("kal meeting hai, please confirm karo") or **romanised** (Hindi typed in Latin letters),
and contains typos, emoji and English product names. I have measured none of these, and there is no reason to assume the
FLORES premium carries over to them. Romanised Hindi uses the Latin script, so it may behave nothing like Devanagari Hindi.
Translationese can also be longer or more literal than native writing.

2009 sentences is enough to pin down a *corpus-level* ratio tightly: the paired-bootstrap 95% CIs in A3 are about ±1%. It
is not enough for per-topic or per-length claims. It also measures only the *input* side: it says nothing about how long the
model's *outputs* are in each language, which is where most decode cost sits.

The domain split shows the premium is stable across wikinews, wikivoyage and wikibooks (Llama-3.2 Hindi 2.51–2.54×), but
those three domains are all formal. So this corpus can rank tokenizers and give a first-order cost premium. The numbers
must be re-checked on a sample of real production traffic before they are used for budgeting (see A4).
