
# Dynamic Tuning for HyLIGHT-Cache (MovieLens 1M)

## Files to add to your project
- Put `tune.py` into: `src/tune/tune.py` (create the `tune/` folder if it doesn't exist).

## YAML additions (configs/experiment.yaml)
```yaml
tuning:
  thresholds: [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
  ensemble_types: [soft, hard, stacking, gating]
  soft_weight_grid:
    gbdt: [0.5, 1.0, 1.5, 2.0]
    nb:   [0.5, 1.0, 1.5]
    elm:  [0.5, 1.0, 1.5, 2.0]
  objective: byte_hit_rate   # or hit_rate or avg_latency_ms
  size_aware:
    alpha: [0.0, 0.2, 0.5]
  calibrate:
    enabled: true
```

## How to run
```bash
# 1) Generate features/labels (already done for MovieLens 1M):
# python -m src.features.featurize --in data/raw/ml1m.csv --out data/processed/ml1m_features.csv --cfg configs/experiment.yaml
# python -m src.labeling.relaxed_belady --in data/raw/ml1m.csv --out data/labels/ml1m_labels.csv --lookahead 5000

# 2) Run tuner (will train base models internally, sweep ensembles & thresholds on VAL, then test best):
python -m src.tune.tune --cfg configs/experiment.yaml   --features data/processed/ml1m_features.csv   --labels data/labels/ml1m_labels.csv   --outdir results/tuning_ml1m/
```

Outputs:
- `results/tuning_ml1m/tuning_leaderboard_val.csv` — semua kandidat (ensemble, weights, alpha, threshold) + metrik VAL dan skor objektif.
- `results/tuning_ml1m/best_config.json` — kandidat terbaik (VAL) beserta parameternya.
- `results/tuning_ml1m/tuning_test_best.csv` — metrik TEST untuk kandidat terbaik.
