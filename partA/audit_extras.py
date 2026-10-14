#!/usr/bin/env python3
"""
audit_extras.py -- the A2 experiments that don't fit the one-switch ablation table.

  python partA/audit_extras.py nfc          # does NFC change token counts on RAW FLORES text?
  python partA/audit_extras.py lower        # which English words get MORE gpt2 tokens when lowercased?
  python partA/audit_extras.py spaces       # empty "words" from split(" ") per language
  python partA/audit_extras.py bootstrap    # how stable is 5.89x with 10 lines vs 2009 lines?
  python partA/audit_extras.py special      # add_special_tokens=False: harmless or not?
  python partA/audit_extras.py seed         # is random.seed(1337) doing anything?
  python partA/audit_extras.py crash        # tiktoken special-token text crashes v0
  python partA/audit_extras.py sample       # are the sample corpora really parallel?
"""

import collections
import importlib.util
import os
import random
import subprocess
import sys
import tarfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT = os.path.join(ROOT, "starter_kit")
CORPUS = os.path.join(HERE, "corpus")
TARBALL = os.path.join(HERE, "data", "flores200_dataset.tar.gz")


def v0():
    spec = importlib.util.spec_from_file_location("fertility_v0", os.path.join(KIT, "fertility.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


# ------------------------------------------------------------------ nfc
def exp_nfc():
    from fertility_v1 import load_tokenizer

    raw = {}
    with tarfile.open(TARBALL, "r:gz") as tar:
        for short, code in [("hin", "hin_Deva"), ("ben", "ben_Beng"), ("kan", "kan_Knda"), ("mal", "mal_Mlym")]:
            txt = []
            for split in ("dev", "devtest"):
                m = next(m for m in tar.getmembers() if m.name.endswith(f"{split}/{code}.{split}"))
                txt += [l.strip() for l in tar.extractfile(m).read().decode("utf-8").split("\n") if l.strip()]
            raw[short] = txt
    for spec in ["gpt2", "hf:unsloth/Llama-3.2-1B", "hf:sarvamai/sarvam-1"]:
        enc, _, _ = load_tokenizer(spec)
        print(f"--- {spec}")
        print(f"{'lang':<6}{'lines changed':>14}{'tokens raw':>12}{'tokens NFC':>12}{'delta':>9}")
        for lang, txt in raw.items():
            nfc = [unicodedata.normalize("NFC", l) for l in txt]
            changed = sum(a != b for a, b in zip(txt, nfc))
            t_raw = sum(len(enc(l)) for l in txt)
            t_nfc = sum(len(enc(l)) for l in nfc)
            print(f"{lang:<6}{changed:>14}{t_raw:>12}{t_nfc:>12}{(t_nfc / t_raw - 1) * 100:>+8.2f}%")
    # what does NFC actually change?
    ex = collections.Counter()
    for txt in raw.values():
        for l in txt:
            n = unicodedata.normalize("NFC", l)
            if n != l:
                for c in set(l) - set(n):
                    ex[f"U+{ord(c):04X} {unicodedata.name(c, '?')}"] += 1
    print("\ncode points removed by NFC (top 5, count = lines):")
    for k, v in ex.most_common(5):
        print(f"  {v:>5}  {k}")


# ------------------------------------------------------------------ lower
def exp_lower():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    words = collections.Counter()
    for l in lines(os.path.join(CORPUS, "eng.txt")):
        for w in l.split():
            words[w] += 1
    more = less = same = 0
    diff = collections.Counter()
    for w, c in words.items():
        a, b = len(enc.encode(" " + w)), len(enc.encode(" " + w.lower()))
        if b > a:
            more += c
            diff[w] += (b - a) * c
        elif b < a:
            less += c
        else:
            same += c
    print("gpt2, English FLORES words (with leading space, as they appear mid-sentence):")
    print(f"  lowercasing ADDS tokens to    {more:>6} word occurrences")
    print(f"  lowercasing REMOVES tokens in {less:>6} word occurrences")
    print(f"  no change                     {same:>6} word occurrences")
    print("\n  biggest contributors (word: tokens original -> lowercased):")
    for w, _ in diff.most_common(10):
        print(f"    {w!r}: {len(enc.encode(' ' + w))} -> {len(enc.encode(' ' + w.lower()))}")
    s = lines(os.path.join(KIT, "corpus_sample", "eng_sample.txt"))
    print("\n  sample-corpus English lines, gpt2 tokens original vs lowercased:")
    for l in s:
        a, b = len(enc.encode(l)), len(enc.encode(l.lower()))
        if a != b:
            print(f"    {a:>3} -> {b:<3} {l}")


# ------------------------------------------------------------------ spaces
def exp_spaces():
    print(f"{'lang':<6}{'split(\" \") words':>18}{'split() words':>15}{'empty words':>13}{'inflation':>11}")
    for f in sorted(os.listdir(CORPUS)):
        if not f.endswith(".txt"):
            continue
        L = lines(os.path.join(CORPUS, f))
        a = sum(len(l.split(" ")) for l in L)
        b = sum(len(l.split()) for l in L)
        print(f"{f[:-4]:<6}{a:>18}{b:>15}{a - b:>13}{(a / b - 1) * 100:>+10.2f}%")
    print("\nstarter_kit sample:")
    for f in ("eng_sample.txt", "hin_sample.txt"):
        L = lines(os.path.join(KIT, "corpus_sample", f))
        a = sum(len(l.split(" ")) for l in L)
        b = sum(len(l.split()) for l in L)
        print(f"  {f}: split(' ') {a}  split() {b}  -> {a - b} empty word(s)")


# ------------------------------------------------------------------ bootstrap
def exp_bootstrap():
    m = v0()
    enc = m.load_tokenizer("gpt2")

    def per_line(path):
        out = []
        for l in m.read_lines(path):
            l = l.lower()
            out.append(len(enc(l)) / len(l.split(" ")))
        return out

    for name, e, h in [
        ("sample (10 lines, NOT parallel)", os.path.join(KIT, "corpus_sample", "eng_sample.txt"),
         os.path.join(KIT, "corpus_sample", "hin_sample.txt")),
        ("FLORES (2009 lines)", os.path.join(CORPUS, "eng.txt"), os.path.join(CORPUS, "hin.txt")),
    ]:
        fe, fh = per_line(e), per_line(h)
        point = (sum(fh) / len(fh)) / (sum(fe) / len(fe))
        rng = random.Random(0)
        vals = []
        for _ in range(5000):
            se = [fe[rng.randrange(len(fe))] for _ in fe]  # independent resampling: the files
            sh = [fh[rng.randrange(len(fh))] for _ in fh]  # are treated as two separate samples
            vals.append((sum(sh) / len(sh)) / (sum(se) / len(se)))
        vals.sort()
        print(f"{name:<34} v0 ratio {point:.2f}x   95% bootstrap CI [{vals[125]:.2f}, {vals[4874]:.2f}]"
              f"   (width {vals[4874] - vals[125]:.2f})")
    print("\nper-line v0 fertility on the sample (shows the spread a 10-line mean hides):")
    print("  eng:", [round(x, 2) for x in per_line(os.path.join(KIT, "corpus_sample", "eng_sample.txt"))])
    print("  hin:", [round(x, 2) for x in per_line(os.path.join(KIT, "corpus_sample", "hin_sample.txt"))])


# ------------------------------------------------------------------ special tokens
def exp_special():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    for f in ("eng", "hin"):
        L = lines(os.path.join(CORPUS, f + ".txt"))
        a = sum(len(tok.encode(l, add_special_tokens=False)) for l in L)
        b = sum(len(tok.encode(l, add_special_tokens=True)) for l in L)
        print(f"xlm-roberta {f}: add_special_tokens=False {a}  True {b}  "
              f"(+{b - a} = {(b - a) / len(L):.0f} per line, +{(b / a - 1) * 100:.1f}%)")
    print("v0 uses add_special_tokens=False -> counts only the text's own tokens. Correct as written.")


# ------------------------------------------------------------------ seed
def exp_seed():
    src = open(os.path.join(KIT, "fertility.py"), encoding="utf-8").read()
    uses = [i + 1 for i, l in enumerate(src.splitlines()) if "random" in l]
    print(f"lines of fertility.py mentioning 'random': {uses}  (import + seed only, never called)")
    outs = set()
    for seed in ("1337", "0", "42"):
        patched = src.replace("random.seed(1337)", f"random.seed({seed})")
        r = subprocess.run([sys.executable, "-c", patched, "--corpus",
                            f"eng={os.path.join(KIT, 'corpus_sample', 'eng_sample.txt')}",
                            "--corpus", f"hin={os.path.join(KIT, 'corpus_sample', 'hin_sample.txt')}"],
                           capture_output=True, text=True, encoding="utf-8")
        outs.add(r.stdout)
        print(f"seed {seed:>4}: {r.stdout.strip().splitlines()[-1]}")
    print(f"distinct outputs across seeds: {len(outs)}  -> seed is dead code, harmless")


# ------------------------------------------------------------------ crash
def exp_crash():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("The model emitted <|endoftext|> and stopped.\n")
        path = f.name
    r = subprocess.run([sys.executable, os.path.join(KIT, "fertility.py"), "--corpus", f"eng={path}"],
                       capture_output=True, text=True, encoding="utf-8")
    print("v0 exit code:", r.returncode)
    print("v0 stderr (last line):", (r.stderr.strip().splitlines() or ["<none>"])[-1])
    os.unlink(path)


# ------------------------------------------------------------------ sample parallelism
def exp_sample():
    e = lines(os.path.join(KIT, "corpus_sample", "eng_sample.txt"))
    h = lines(os.path.join(KIT, "corpus_sample", "hin_sample.txt"))
    for i, (a, b) in enumerate(zip(e, h), 1):
        print(f"{i:>2} | {a}\n   | {b}")


EXPS = dict(nfc=exp_nfc, lower=exp_lower, spaces=exp_spaces, bootstrap=exp_bootstrap,
            special=exp_special, seed=exp_seed, crash=exp_crash, sample=exp_sample)

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    if len(sys.argv) != 2 or sys.argv[1] not in EXPS:
        sys.exit(__doc__)
    EXPS[sys.argv[1]]()
