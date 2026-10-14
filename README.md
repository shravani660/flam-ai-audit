# The Audit: submission

| part | read this | code | raw output |
|---|---|---|---|
| A1 corpus | `partA/A1_corpus.md` | `partA/get_corpus.py` | `partA/results/01_corpus.txt` |
| A2 audit | `partA/A2_audit.md` | `partA/audit_ablation.py`, `partA/audit_extras.py` | `partA/results/02_*`, `03_*` |
| A3 analysis | `partA/A3_analysis.md` | `partA/fertility_v1.py` | `partA/results/04_main.txt`, `main.csv` |
| A4 memo | `partA/A4_memo.md` | | |
| B1–B4 | `partB/partB.md` | `partB/capacity.py` | `partB/capacity_output.txt` |
| C memo | `partC/memo.md` | | |
| process | `NOTEBOOK.md`, `AI_USAGE.md` | `run_all.sh` | |

`starter_kit/` is the original kit, unmodified (only the macOS `.DS_Store` was removed).

## Reproduce everything
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (Linux/mac: .venv/bin/pip)
PY=.venv/Scripts/python bash run_all.sh                                  # from this folder
```
`get_corpus.py` downloads FLORES-200 (about 25 MB) on first run. The tokenizers download from Hugging Face; none of them
are gated. No GPU is needed: Part B is arithmetic on the provided log.

## Single commands worth knowing
```bash
python partA/audit_ablation.py --pair sample            # one-switch-at-a-time ablation of v0
python partA/audit_extras.py <nfc|lower|spaces|bootstrap|seed|special|crash|sample>
python partA/fertility_v1.py --tokenizer gpt2 --langs eng,hin
python partA/fertility_v1.py --tokenizer gpt2 --corpus-dir some/dir --file-pattern {lang}.txt --langs eng,hin   # files MUST be translations line-by-line
python partA/quick_count.py "any text" "more text" --tokenizer gpt2 --tokenizer hf:sarvamai/sarvam-1   # live checks
python partB/capacity.py
```
