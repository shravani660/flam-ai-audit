#!/usr/bin/env bash
# Re-runs every experiment and rewrites every results file. Run from submission/.
#   PY=../.venv/Scripts/python bash run_all.sh      (Windows venv)
#   PY=python bash run_all.sh                       (any env with requirements.txt installed)
set -euo pipefail
PY=${PY:-python}
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")";; esac
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
R=partA/results
mkdir -p $R

echo "[0] baseline: original script, original data"
(cd starter_kit && "$PY" fertility.py --corpus eng=corpus_sample/eng_sample.txt \
   --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2) | tee $R/00_baseline_v0.txt

echo "[1] A1 corpus"
"$PY" partA/get_corpus.py | tee $R/01_corpus.txt

echo "[2] A2 ablation"
"$PY" partA/audit_ablation.py --pair sample | tee $R/02_ablation_sample.txt
"$PY" partA/audit_ablation.py --pair flores | tee $R/02_ablation_flores.txt

echo "[3] A2 extras"
for e in sample spaces lower bootstrap seed special crash nfc; do
  "$PY" partA/audit_extras.py $e 2>&1 | grep -v -i "pytorch" | tee $R/03_extra_$e.txt
done

echo "[4] A3 corrected analysis"
"$PY" partA/fertility_v1.py --tokenizer gpt2 --tokenizer hf:unsloth/Llama-3.2-1B \
  --tokenizer hf:sarvamai/sarvam-1 --tokenizer hf:FacebookAI/xlm-roberta-base \
  --langs eng,hin,kan,tam,tel,mal,ben,mar --by-domain --out $R/main 2>&1 | grep -v -i "pytorch" | tee $R/04_main.txt

echo "[5] Part B"
"$PY" partB/capacity.py | tee partB/capacity_output.txt
echo "done"
