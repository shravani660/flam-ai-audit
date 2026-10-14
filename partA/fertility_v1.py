#!/usr/bin/env python3
"""
fertility_v1.py -- corrected cross-language tokenizer cost benchmark.

Differences from fertility.py (v0) -- each one justified in A2_audit.md:
  * no lowercasing (tokenize exactly what users send)
  * words = str.split() (any whitespace run), never split(" ")
  * aggregate as ratio of totals (sum tokens / sum denominator), not mean of per-line ratios
  * several denominators reported side by side:
        per parallel sentence  -> the "same content" denominator
        per whitespace word, per grapheme cluster (\\X), per UTF-8 byte
  * premium vs English = tokens(lang) / tokens(eng) over the SAME parallel sentences,
    with a paired bootstrap 95% CI (resample sentence ids, same ids for both languages)
  * reports the [UNK] rate, because a tokenizer that maps text to [UNK] looks "efficient"
  * input must be parallel: every corpus file must have the same number of lines

Usage:
  python partA/fertility_v1.py --tokenizer gpt2 --tokenizer hf:sarvamai/sarvam-1 \
      --langs eng,hin,kan,tam,tel,mal --out partA/results/main
  python partA/fertility_v1.py --tokenizer gpt2 --corpus-dir starter_kit/corpus_sample \
      --file-pattern {lang}_sample.txt --langs eng,hin        # any parallel files

Tokenizer specs: tiktoken encodings (gpt2, o200k_base, ...) or hf:<repo_id>.
"""

import argparse
import csv
import os
import random
import unicodedata

import regex

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- tokenizers
def load_tokenizer(spec):
    """Return (encode_fn, unk_id_or_None, vocab_size)."""
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec[3:])
        return (lambda s: tok.encode(s, add_special_tokens=False)), tok.unk_token_id, len(tok)
    import tiktoken

    enc = tiktoken.get_encoding(spec)
    return (lambda s: enc.encode(s, disallowed_special=())), None, enc.n_vocab


# ---------------------------------------------------------------- corpus
def read_parallel(corpus_dir, pattern, langs):
    data = {}
    for lang in langs:
        path = os.path.join(corpus_dir, pattern.format(lang=lang))
        with open(path, "r", encoding="utf-8") as f:
            # keep empty lines as placeholders so alignment is never broken
            data[lang] = [unicodedata.normalize("NFC", l.rstrip("\n").strip()) for l in f]
        while data[lang] and data[lang][-1] == "":
            data[lang].pop()
    n = {len(v) for v in data.values()}
    if len(n) != 1:
        raise SystemExit(f"corpora are not parallel: { {k: len(v) for k, v in data.items()} }")
    return data


def line_stats(line):
    return dict(
        words=len(line.split()),
        graphemes=len(regex.findall(r"\X", line)),
        codepoints=len(line),
        bytes=len(line.encode("utf-8")),
    )


# ---------------------------------------------------------------- stats
def paired_bootstrap(a, b, reps=2000, seed=0):
    """95% CI of sum(a)/sum(b) resampling sentence indices jointly."""
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(a[i] for i in idx) / sum(b[i] for i in idx))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="append", required=True)
    ap.add_argument("--langs", default="eng,hin,kan,tam,tel,mal")
    ap.add_argument("--base", default="eng")
    ap.add_argument("--corpus-dir", default=os.path.join(HERE, "corpus"))
    ap.add_argument("--file-pattern", default="{lang}.txt")
    ap.add_argument("--out", default=None, help="prefix for CSV outputs")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--by-domain", action="store_true", help="premium split by FLORES domain")
    args = ap.parse_args()

    langs = args.langs.split(",")
    data = read_parallel(args.corpus_dir, args.file_pattern, langs)
    n_sent = len(data[langs[0]])
    keep = [i for i in range(n_sent) if all(data[l][i] for l in langs)]  # drop ids empty in ANY language

    # denominators are tokenizer-independent
    den = {l: [line_stats(data[l][i]) for i in keep] for l in langs}

    print(f"corpus: {args.corpus_dir}  sentences used: {len(keep)} "
          f"(line counts match; that content is parallel is ASSUMED, not checked)\n")
    print("denominators per sentence (mean):")
    print(f"{'lang':<6}{'words':>8}{'graph':>8}{'cpts':>8}{'bytes':>8}{'bytes/graph':>13}")
    for l in langs:
        s = {k: sum(d[k] for d in den[l]) / len(keep) for k in den[l][0]}
        print(f"{l:<6}{s['words']:>8.1f}{s['graphemes']:>8.1f}{s['codepoints']:>8.1f}{s['bytes']:>8.1f}"
              f"{s['bytes'] / s['graphemes']:>13.2f}")
    print()

    rows = []
    for spec in args.tokenizer:
        encode, unk_id, vsize = load_tokenizer(spec)
        toks = {}
        unk = {}
        for l in langs:
            ids = [encode(data[l][i]) for i in keep]
            toks[l] = [len(x) for x in ids]
            unk[l] = sum(x.count(unk_id) for x in ids) if unk_id is not None else 0

        print(f"=== tokenizer: {spec}  (vocab {vsize})")
        print(f"{'lang':<6}{'tok/sent':>9}{'tok/word':>9}{'tok/graph':>10}{'tok/byte':>9}"
              f"{'premium vs ' + args.base:>17}{'95% CI':>16}{'word-ratio':>11}{'unk%':>7}")
        T_base = toks[args.base]
        W_base = sum(d["words"] for d in den[args.base])
        for l in langs:
            T = sum(toks[l])
            W = sum(d["words"] for d in den[l])
            G = sum(d["graphemes"] for d in den[l])
            B = sum(d["bytes"] for d in den[l])
            premium = T / sum(T_base)
            lo, hi = paired_bootstrap(toks[l], T_base, args.reps) if l != args.base else (1.0, 1.0)
            word_ratio = (T / W) / (sum(T_base) / W_base)  # what v0 would call "Nx the fertility"
            unk_pct = 100 * unk[l] / T
            print(f"{l:<6}{T / len(keep):>9.1f}{T / W:>9.2f}{T / G:>10.3f}{T / B:>9.3f}"
                  f"{premium:>16.2f}x{f'[{lo:.2f}, {hi:.2f}]':>16}{word_ratio:>10.2f}x{unk_pct:>7.2f}")
            rows.append(dict(tokenizer=spec, vocab=vsize, lang=l, sentences=len(keep), tokens=T,
                             words=W, graphemes=G, bytes=B, tok_per_sent=round(T / len(keep), 3),
                             tok_per_word=round(T / W, 4), tok_per_grapheme=round(G and T / G, 4),
                             tok_per_byte=round(T / B, 4), premium_vs_base=round(premium, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             word_fertility_ratio=round(word_ratio, 4), unk_pct=round(unk_pct, 3)))
        if args.by_domain:
            dom_path = os.path.join(args.corpus_dir, "domains.tsv")
            with open(dom_path, encoding="utf-8") as f:
                next(f)
                dom = [r.split("\t")[2] for r in f]
            dom = [dom[i] for i in keep]
            print(f"  premium vs {args.base} by FLORES domain:")
            print(f"  {'lang':<6}" + "".join(f"{d:>12}" for d in sorted(set(dom))))
            for l in langs:
                cells = []
                for d in sorted(set(dom)):
                    idx = [k for k, x in enumerate(dom) if x == d]
                    cells.append(sum(toks[l][k] for k in idx) / sum(T_base[k] for k in idx))
                print(f"  {l:<6}" + "".join(f"{c:>11.2f}x" for c in cells))
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}.csv")


if __name__ == "__main__":
    main()
