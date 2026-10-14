#!/usr/bin/env python3
"""
quick_count.py -- count tokens / words / graphemes / bytes for text typed on the command line.
For live checks: python partA/quick_count.py "किताबें  अलमारी में रखी हैं।" --tokenizer gpt2 --tokenizer hf:sarvamai/sarvam-1
"""

import argparse
import unicodedata

import regex

from fertility_v1 import load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("text", nargs="+")
ap.add_argument("--tokenizer", action="append", default=None)
args = ap.parse_args()

for spec in args.tokenizer or ["gpt2"]:
    enc, _, _ = load_tokenizer(spec)
    print(f"--- {spec}")
    for t in args.text:
        t = unicodedata.normalize("NFC", t)
        n = len(enc(t))
        w, g, b = len(t.split()), len(regex.findall(r"\X", t)), len(t.encode("utf-8"))
        print(f"{n:>4} tok | {w} words ({n / w:.2f}/w) | split(' ') words {len(t.split(' '))} | "
              f"{g} graphemes | {len(t)} code points | {b} bytes | {t}")
