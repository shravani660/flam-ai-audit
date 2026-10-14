#!/usr/bin/env python3
"""
get_corpus.py -- build the Part A eval corpus from FLORES-200 (dev + devtest).

Downloads the official FLORES-200 tarball (CC-BY-SA 4.0, Meta/NLLB) once,
extracts the languages we care about, and writes one sentence per line:

    partA/corpus/<lang>.txt      (line i is the same sentence in every language)
    partA/corpus/domains.tsv     (line i -> source domain + topic, from FLORES metadata)

Preprocessing is deliberately minimal (see A1_corpus.md):
  * strip trailing newline only
  * NFC normalisation (no-op check is reported, not assumed)
  * no lowercasing, no space collapsing, no filtering
Then it verifies the corpus really is parallel (same line count everywhere).

Usage:  python partA/get_corpus.py
"""

import os
import sys
import tarfile
import unicodedata
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "corpus")
URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
TARBALL = os.path.join(DATA, "flores200_dataset.tar.gz")

# short code -> FLORES code
LANGS = {
    "eng": "eng_Latn",
    "hin": "hin_Deva",
    "kan": "kan_Knda",  # Dravidian
    "tam": "tam_Taml",  # Dravidian
    "tel": "tel_Telu",  # Dravidian
    "mal": "mal_Mlym",  # Dravidian
    "ben": "ben_Beng",  # not required for A; included because Part C targets it
    "mar": "mar_Deva",  # same
}
SPLITS = ["dev", "devtest"]


def member(tar, suffix):
    """Archive paths start with './' -- match on suffix instead of exact name."""
    for m in tar.getmembers():
        if m.name.endswith(suffix):
            return m
    raise KeyError(suffix)


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(TARBALL):
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, TARBALL)

    with tarfile.open(TARBALL, "r:gz") as tar:
        # domain metadata (same row order as the sentence files)
        domains = []
        for split in SPLITS:
            raw = tar.extractfile(member(tar, f"metadata_{split}.tsv")).read().decode("utf-8")
            rows = raw.splitlines()
            header, rows = rows[0].split("\t"), rows[1:]
            for r in rows:
                d = dict(zip(header, r.split("\t")))
                domains.append((split, d.get("domain", "?"), d.get("topic", "?")))

        counts = {}
        for short, code in LANGS.items():
            lines, changed_by_nfc = [], 0
            for split in SPLITS:
                raw = tar.extractfile(member(tar, f"{split}/{code}.{split}")).read().decode("utf-8")
                for line in raw.split("\n"):
                    if line == "":
                        continue
                    nfc = unicodedata.normalize("NFC", line)
                    changed_by_nfc += nfc != line
                    lines.append(nfc)
            counts[short] = len(lines)
            with open(os.path.join(OUT, f"{short}.txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
            print(f"{short} ({code}): {len(lines)} sentences, NFC changed {changed_by_nfc} lines")

    with open(os.path.join(OUT, "domains.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("line\tsplit\tdomain\ttopic\n")
        for i, (s, d, t) in enumerate(domains, 1):
            f.write(f"{i}\t{s}\t{d}\t{t}\n")

    n = set(counts.values())
    if len(n) != 1 or n.pop() != len(domains):
        sys.exit(f"NOT PARALLEL: {counts}, metadata rows {len(domains)}")
    print(f"OK: parallel, {len(domains)} sentences x {len(LANGS)} languages")


if __name__ == "__main__":
    main()
