#!/usr/bin/env bash
set -e
# ensure folders exist
mkdir -p data/raw data/processed data/labels models results

echo "[1/6] Generate synthetic trace"
python -m src.parse.synthetic_trace --out data/raw/synth_trace.csv --n 20000 --seed 42
echo "[2/6] Featurize"
python -m src.features.featurize --in data/raw/synth_trace.csv --out data/processed/synth_features.csv
echo "[3/6] Label (Relaxed Belady)"
python -m src.labeling.relaxed_belady --in data/raw/synth_trace.csv --out data/labels/synth_labels.csv --capacity_bytes 134217728
echo "[4/6] Train models (Ensemble + LR-SGD with decay)"
python -m src.ml.train --cfg configs/experiment.yaml --features data/processed/synth_features.csv --labels data/labels/synth_labels.csv --out models/
echo "[5/6] Evaluate (classification + cache replay)"
python -m src.ml.eval --cfg configs/experiment.yaml --features data/processed/synth_features.csv --labels data/labels/synth_labels.csv --models_dir models/ --out results/
echo "[6/6] Done. See results/"
