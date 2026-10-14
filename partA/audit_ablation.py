#!/usr/bin/env python3
"""
audit_ablation.py -- isolate each suspected flaw in starter_kit/fertility.py.

Method: re-implement fertility.py's analyze() with every questionable choice
behind a switch. The all-defaults configuration must reproduce the original
script bit-for-bit (checked by importing the original and comparing). Then we
flip ONE switch at a time and record how eng / hin / the eng->hin ratio move.

Switches (original behaviour first):
  lower   : True  | False            -- line.lower() before tokenizing
  split   : space | ws               -- line.split(" ") vs line.split()
  agg     : lines | totals           -- mean of per-line ratios vs ratio of sums
  nfc     : True  | False            -- unicodedata NFC in read_lines()
  char    : cp    | graph | byte     -- len(line) code points vs graphemes vs UTF-8 bytes

Usage:
  python partA/audit_ablation.py --pair sample    # starter_kit corpus_sample (10 lines)
  python partA/audit_ablation.py --pair flores    # partA/corpus eng/hin (2009 lines)
  python partA/audit_ablation.py --pair flores --tokenizer hf:sarvamai/sarvam-1
"""

import argparse
import importlib.util
import os
import unicodedata

import regex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.join(ROOT, "starter_kit")

PAIRS = {
    "sample": (os.path.join(KIT, "corpus_sample", "eng_sample.txt"),
               os.path.join(KIT, "corpus_sample", "hin_sample.txt")),
    "flores": (os.path.join(HERE, "corpus", "eng.txt"),
               os.path.join(HERE, "corpus", "hin.txt")),
}

DEFAULT = dict(lower=True, split="space", agg="lines", nfc=True, char="cp")
VARIANTS = [
    ("baseline (original script)", {}),
    ("no lowercasing", dict(lower=False)),
    ("split() instead of split(' ')", dict(split="ws")),
    ("ratio of totals, not mean of lines", dict(agg="totals")),
    ("no NFC", dict(nfc=False)),
    ("graphemes instead of code points", dict(char="graph")),
    ("UTF-8 bytes instead of code points", dict(char="byte")),
    ("all code fixes (lower off, ws, totals, graph)",
     dict(lower=False, split="ws", agg="totals", char="graph")),
]


def load_original():
    spec = importlib.util.spec_from_file_location("fertility_v0", os.path.join(KIT, "fertility.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_lines(path, nfc):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if nfc:
                line = unicodedata.normalize("NFC", line)
            out.append(line)
    return out


def n_chars(line, mode):
    if mode == "cp":
        return len(line)
    if mode == "graph":
        return len(regex.findall(r"\X", line))
    if mode == "byte":
        return len(line.encode("utf-8"))
    raise ValueError(mode)


def analyze(lines, encode, lower, split, agg, char, **_):
    fert, tpc = [], []
    T = W = C = 0
    for line in lines:
        if lower:
            line = line.lower()
        t = len(encode(line))
        w = len(line.split(" ") if split == "space" else line.split())
        c = n_chars(line, char)
        fert.append(t / w)
        tpc.append(t / c)
        T, W, C = T + t, W + w, C + c
    if agg == "lines":
        return sum(fert) / len(fert), sum(tpc) / len(tpc)
    return T / W, T / C


def run(pair, encode, cfg):
    e_path, h_path = PAIRS[pair]
    res = {}
    for lang, path in (("eng", e_path), ("hin", h_path)):
        res[lang] = analyze(read_lines(path, cfg["nfc"]), encode, **cfg)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=PAIRS, default="sample")
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    orig = load_original()
    encode = orig.load_tokenizer(args.tokenizer)

    # control: our defaults must equal the original script exactly
    base = run(args.pair, encode, DEFAULT)
    for lang, path in zip(("eng", "hin"), PAIRS[args.pair]):
        o = orig.analyze(orig.read_lines(path), encode)
        assert o == base[lang], f"control mismatch for {lang}: {o} vs {base[lang]}"
    print(f"control OK: defaults reproduce starter_kit/fertility.py exactly "
          f"(pair={args.pair}, tokenizer={args.tokenizer})\n")

    hdr = f"{'variant':<48}{'eng t/w':>9}{'hin t/w':>9}{'ratio':>8}{'d ratio':>9}   " \
          f"{'eng t/c':>8}{'hin t/c':>8}{'ratio':>8}"
    print(hdr)
    print("-" * len(hdr))
    r0 = base["hin"][0] / base["eng"][0]
    for name, delta in VARIANTS:
        cfg = {**DEFAULT, **delta}
        r = run(args.pair, encode, cfg)
        rw = r["hin"][0] / r["eng"][0]
        rc = r["hin"][1] / r["eng"][1]
        print(f"{name:<48}{r['eng'][0]:>9.3f}{r['hin'][0]:>9.3f}{rw:>8.2f}{(rw / r0 - 1) * 100:>+8.1f}%   "
              f"{r['eng'][1]:>8.3f}{r['hin'][1]:>8.3f}{rc:>8.2f}")


if __name__ == "__main__":
    main()
