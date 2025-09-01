import argparse, yaml, pandas as pd, numpy as np, json, os
from .models import make_lr_sgd_with_decay, make_gbdt, make_nb, ELMClassifier, SoftVotingEnsemble

BASE_COLS = ['row_id','ts','key','size_bytes']

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def split_data(df_feat, df_lbl, split_cfg, feature_cols):
    df = df_feat.merge(df_lbl, on=['row_id'], how='inner').dropna()
    df = df.sample(frac=1.0, random_state=split_cfg.get('random_state', 123)).reset_index(drop=True)
    n = len(df); n_train = int(split_cfg['train']*n); n_val = int(split_cfg['val']*n)
    parts = {'train': df.iloc[:n_train], 'val': df.iloc[n_train:n_train+n_val], 'test': df.iloc[n_train+n_val:]}
    out={}
    for name, part in parts.items():
        X = part[feature_cols].values
        y = part['label'].astype(int).values
        out[name] = (X, y, part)
    return out

def _cast_sgd_cfg(d):
    out = dict(d)
    out['eta0']     = float(out['eta0']);  out['power_t']  = float(out['power_t'])
    out['alpha']    = float(out['alpha']); out['max_iter'] = int(out['max_iter'])
    return out

def _cast_elm_cfg(d):
    out = dict(d)
    out['n_hidden']   = int(out['n_hidden'])
    out['reg_lambda'] = float(out['reg_lambda'])
    out['activation'] = str(out.get('activation','relu'))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg); os.makedirs(args.out, exist_ok=True)
    df_feat = pd.read_csv(args.features); df_lbl  = pd.read_csv(args.labels)

    feature_cols = [c for c in df_feat.columns if c not in BASE_COLS]
    if not feature_cols: raise ValueError("No feature columns found in features CSV.")
    with open(os.path.join(args.out, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    splits = split_data(df_feat, df_lbl, cfg['split'], feature_cols)
    (Xtr, ytr, _), (Xv, yv, _), (Xte, yte, _) = splits['train'], splits['val'], splits['test']

    lr_sgd = make_lr_sgd_with_decay(**_cast_sgd_cfg(cfg['sgd_decay']))
    gbdt   = make_gbdt(cfg['lightgbm'])
    nb     = make_nb(cfg['naive_bayes']['type'])
    elm    = ELMClassifier(**_cast_elm_cfg(cfg['elm']))

    lr_sgd.fit(Xtr, ytr); gbdt.fit(Xtr, ytr); nb.fit(Xtr, ytr); elm.fit(Xtr, ytr)
    models   = {'lightgbm': gbdt, 'naive_bayes': nb, 'elm': elm}
    ensemble = SoftVotingEnsemble(models, cfg['ensemble_weights']).fit(Xtr, ytr)

    import joblib
    joblib.dump(lr_sgd, os.path.join(args.out, 'lr_sgd_decay.joblib'))
    joblib.dump(gbdt,   os.path.join(args.out, 'gbdt.joblib'))
    joblib.dump(nb,     os.path.join(args.out, 'naive_bayes.joblib'))
    joblib.dump(elm,    os.path.join(args.out, 'elm.joblib'))
    joblib.dump(ensemble, os.path.join(args.out, 'ensemble_soft.joblib'))
    print('Models saved to', args.out)

if __name__ == '__main__':
    main()