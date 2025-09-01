import argparse, pandas as pd, numpy as np, yaml, re
from ..common.schema import TraceCols

def _load_cfg(path):
    if not path: return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def _sanitize(s: str):
    import re
    return re.sub(r'[^0-9A-Za-z]+', '_', str(s)).strip('_')[:64]

def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # simpan index asli sebagai row_id SEBELUM operasi apa pun
    df = df.copy()
    df['row_id'] = df.index.astype(int)

    # sortir untuk stabilitas fitur berbasis waktu
    df.sort_values(TraceCols.ts, inplace=True)

    fcfg = (cfg.get('features') or {}) if isinstance(cfg, dict) else {}
    log_size      = bool(fcfg.get('log_size', True))
    use_recency   = bool(fcfg.get('use_recency', True))
    use_freq      = bool(fcfg.get('use_freq', True))
    hour_from_ts  = bool(fcfg.get('hour_from_ts', True))
    use_ttl       = bool(fcfg.get('use_ttl', True))
    extra_num     = list(fcfg.get('extra_numeric', []) or [])
    extra_cat     = list(fcfg.get('extra_categorical', []) or [])
    top_k         = int(fcfg.get('categorical_top_k', 20))

    out = pd.DataFrame(index=df.index)
    base_keep = ['row_id', TraceCols.ts, TraceCols.key, TraceCols.size]  # identitas + ukuran

    # ----- fitur dasar -----
    if log_size:
        out['log_size'] = np.log1p(df[TraceCols.size])

    if use_recency:
        prev_ts = df.groupby(TraceCols.key)[TraceCols.ts].shift(1)
        rec = df[TraceCols.ts] - prev_ts
        fill_val = rec.max(skipna=True)
        if pd.isna(fill_val): fill_val = 0.0
        out['recency'] = rec.fillna(fill_val)

    if use_freq:
        out['freq'] = df.groupby(TraceCols.key).cumcount() + 1

    if hour_from_ts:
        out['hour'] = (df[TraceCols.ts] // 3600) % 24

    if use_ttl:
        ttl = df[TraceCols.ttl] if TraceCols.ttl in df.columns else -1
        ttl = pd.Series(ttl, index=df.index)
        out['ttl_pos'] = (ttl.astype(float) > 0).astype(int)
        out['ttl_val'] = ttl.astype(float).clip(lower=0)

    # Filter GET jika ada kolom op
    if TraceCols.op in df.columns:
        is_get = (df[TraceCols.op] == 'GET')
        df = df[is_get.values]
        out = out.loc[df.index]

    # ----- fitur numerik ekstra -----
    for col in extra_num:
        if col in df.columns:
            out[f'num_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # ----- fitur kategorikal ekstra (one-hot top-K + OTHER) -----
    for col in extra_cat:
        if col in df.columns:
            vc = df[col].astype(str).value_counts()
            keep_vals = set(vc.head(top_k).index.tolist())
            col_series = df[col].astype(str)
            for v in keep_vals:
                out[f'cat_{col}__{_sanitize(v)}'] = (col_series == v).astype(int)
            out[f'cat_{col}__OTHER'] = (~col_series.isin(keep_vals)).astype(int)

    # rangkai output akhir: identitas + fitur
    out_full = pd.concat([df[base_keep].reset_index(drop=True),
                          out.reset_index(drop=True)], axis=1)

    feat_cols = [c for c in out_full.columns if c not in base_keep]
    if not feat_cols:
        raise ValueError("No features selected/constructed; check your features config.")
    return out_full

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='in_path',  required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    ap.add_argument('--cfg', dest='cfg_path', default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.cfg_path)
    df = pd.read_csv(args.in_path)
    out = compute_features(df, cfg)
    out.to_csv(args.out_path, index=False)

    base_keep = ['row_id', TraceCols.ts, TraceCols.key, TraceCols.size]
    feat_cols = [c for c in out.columns if c not in base_keep]
    print(f"Wrote features: {args.out_path} rows={len(out)} n_features={len(feat_cols)}")

if __name__ == '__main__':
    main()