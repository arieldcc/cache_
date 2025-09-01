import argparse, json, os, yaml, numpy as np, pandas as pd
from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from src.ml.models import make_lr_sgd_with_decay, make_gbdt, make_nb, ELMClassifier

BASE_COLS = ['row_id','ts','key','size_bytes']

class LRUCache:
    def __init__(self, capacity_bytes:int):
        self.capacity=int(capacity_bytes); self.cur=0; self.store={}; self.order=[]
    def _touch(self,k):
        if k in self.store: self.order.remove(k); self.order.append(k)
    def get(self,k):
        hit=k in self.store
        if hit: self._touch(k)
        return hit
    def put(self,k,s):
        if k in self.store: self._touch(k); return
        while self.cur+s>self.capacity and self.order:
            k0=self.order.pop(0); s0=self.store.pop(k0); self.cur-=s0
        if s<=self.capacity:
            self.store[k]=s; self.order.append(k); self.cur+=s

def eval_cache(df_feat_min:pd.DataFrame, prob:np.ndarray, cfg:dict, thr:float, size_alpha:float=0.0)->Dict[str,float]:
    df=df_feat_min.sort_values('ts').reset_index(drop=True)
    t=np.full(len(df), float(thr))
    if size_alpha>0:
        ls=df['log_size'].to_numpy()
        ls=(ls-ls.min())/max(ls.max()-ls.min(),1e-9)
        t=t+float(size_alpha)*ls
    admit=(prob>=t).astype(int)
    cache=LRUCache(cfg['capacity_bytes'])
    hits=reqs=bytes_hit=bytes_req=0; lat=0.0; lh=cfg['latency_ms']['hit']; lm=cfg['latency_ms']['miss']
    for i,r in df.iterrows():
        sz=int(r['size_bytes']); k=r['key']; reqs+=1; bytes_req+=sz
        if cache.get(k): hits+=1; bytes_hit+=sz; lat+=lh
        else:
            lat+=lm
            if admit[i]==1: cache.put(k,sz)
    return {'requests':reqs,'hits':hits,'hit_rate':hits/max(1,reqs),
            'byte_hit_rate':bytes_hit/max(1,bytes_req),'avg_latency_ms':lat/max(1,reqs)}

def load_cfg(p): 
    with open(p,'r') as f: return yaml.safe_load(f)

def split_merge(df_feat, df_lbl, split_cfg, feature_cols):
    df=df_feat.merge(df_lbl[['row_id','label']], on='row_id', how='inner').reset_index(drop=True)
    df=df.sample(frac=1.0, random_state=split_cfg.get('random_state',123)).reset_index(drop=True)
    n=len(df); n_tr=int(split_cfg['train']*n); n_val=int(split_cfg['val']*n)
    parts={'train':df.iloc[:n_tr], 'val':df.iloc[n_tr:n_tr+n_val], 'test':df.iloc[n_tr+n_val:]}
    out={}
    for k,part in parts.items():
        out[k]={'X':part[feature_cols].to_numpy(),
                'y':part['label'].astype(int).to_numpy(),
                'df_min':part[['row_id','ts','key','size_bytes','log_size']].copy()}
    return out

def fit_bases(cfg,X,y):
    m={'gbdt':make_gbdt(cfg['lightgbm']),
       'nb':make_nb(cfg['naive_bayes']['type']),
       'elm':ELMClassifier(**cfg['elm']),
       'lr':make_lr_sgd_with_decay(**cfg['sgd_decay'])}
    for name,model in m.items(): 
        print(f"[fit] {name}", flush=True); model.fit(X,y)
    return m

def proba_all(models,X): 
    out={}
    for k,m in models.items():
        p=m.predict_proba(X)[:,1]
        out[k]=np.clip(p,1e-6,1-1e-6)
    return out

def soft_vote(probs:Dict[str,np.ndarray], w:Dict[str,float]): 
    s=None; tw=0.0
    for k,p in probs.items():
        wk=float(w.get(k,1.0)); s=p*wk if s is None else s+wk*p; tw+=wk
    return np.clip(s/max(tw,1e-9),1e-6,1-1e-6)

def hard_vote(probs:Dict[str,np.ndarray]):
    v=np.stack([(p>=0.5).astype(int) for p in probs.values()], axis=1).mean(axis=1)
    return np.clip(v,1e-6,1-1e-6)

def platt(p_train,y_train):
    lr=LogisticRegression(max_iter=1000); x=p_train.reshape(-1,1)
    lr.fit(x,y_train)
    return lambda p: lr.predict_proba(np.clip(p.reshape(-1,1),1e-6,1-1e-6))[:,1]

def stacking_meta(P_val,y_val):
    clf=LogisticRegression(max_iter=1000); clf.fit(P_val,y_val)
    return lambda P_new: clf.predict_proba(P_new)[:,1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--outdir', required=True)
    args=ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[load] cfg/features/labels", flush=True)
    cfg=load_cfg(args.cfg)
    df_feat=pd.read_csv(args.features); df_lbl=pd.read_csv(args.labels)

    feature_cols=[c for c in df_feat.columns if c not in BASE_COLS]
    if not feature_cols: raise ValueError("No feature columns")
    splits=split_merge(df_feat, df_lbl, cfg['split'], feature_cols)
    Xtr,ytr,df_tr=splits['train']['X'],splits['train']['y'],splits['train']['df_min']
    Xv,yv,df_v  =splits['val']['X'],  splits['val']['y'],  splits['val']['df_min']
    Xte,yte,df_te=splits['test']['X'],splits['test']['y'],splits['test']['df_min']

    print("[fit] base models", flush=True)
    bases=fit_bases(cfg,Xtr,ytr)
    print("[proba] train/val/test", flush=True)
    p_tr=proba_all(bases,Xtr); p_v=proba_all(bases,Xv); p_te=proba_all(bases,Xte)

    # Calibrate (Platt per base)
    tcfg=cfg.get('tuning',{})
    do_cal=tcfg.get('calibrate',{}).get('enabled', True)
    cals={}
    if do_cal:
        print("[calibrate] Platt", flush=True)
        for k in ['gbdt','nb','elm','lr']:
            cals[k]=platt(p_tr[k], ytr)
        p_v={k:cals[k](p_v[k]) for k in p_v}
        p_te={k:cals[k](p_te[k]) for k in p_te}

    thresholds = tcfg.get('thresholds', [0.5,0.6,0.7,0.8,0.85,0.9])
    ensemble_types = tcfg.get('ensemble_types', ['soft','hard','stacking','gating'])
    soft_grid = tcfg.get('soft_weight_grid', {'gbdt':[1.0],'nb':[1.0],'elm':[1.0]})
    size_alphas = tcfg.get('size_aware',{}).get('alpha',[0.0,0.2,0.5])
    objective = tcfg.get('objective','byte_hit_rate')

    print(f"[grid] ensembles={ensemble_types} thr={thresholds} alpha={size_alphas}", flush=True)

    def objv(m):
        return m['byte_hit_rate'] if objective in ('byte_hit_rate','hit_rate') else -m['avg_latency_ms']

    rows=[]
    # SOFT & HARD
    from itertools import product
    if 'soft' in ensemble_types:
        for wg in product(soft_grid.get('gbdt',[1.0]), soft_grid.get('nb',[1.0]), soft_grid.get('elm',[1.0])):
            w={'gbdt':float(wg[0]),'nb':float(wg[1]),'elm':float(wg[2])}
            pv=soft_vote(p_v,w); 
            for a in size_alphas:
                for t in thresholds:
                    m=eval_cache(df_v, pv, cfg, t, a)
                    rows.append({'ensemble':'soft','weights':json.dumps(w),'alpha':a,'threshold':t,
                                 'val_hit_rate':m['hit_rate'],'val_byte_hit_rate':m['byte_hit_rate'],'val_avg_latency_ms':m['avg_latency_ms'],'objective':objv(m)})
    if 'hard' in ensemble_types:
        pv=hard_vote(p_v)
        for a in size_alphas:
            for t in thresholds:
                m=eval_cache(df_v, pv, cfg, t, a)
                rows.append({'ensemble':'hard','weights':'{}','alpha':a,'threshold':t,
                             'val_hit_rate':m['hit_rate'],'val_byte_hit_rate':m['byte_hit_rate'],'val_avg_latency_ms':m['avg_latency_ms'],'objective':objv(m)})

    # STACKING (meta LR pada proba base)
    if 'stacking' in ensemble_types:
        Pv=np.vstack([p_v[k] for k in ['gbdt','nb','elm','lr']]).T
        meta=stacking_meta(Pv,yv)
        pv=meta(Pv)
        for a in size_alphas:
            for t in thresholds:
                m=eval_cache(df_v, pv, cfg, t, a)
                rows.append({'ensemble':'stacking','weights':'{}','alpha':a,'threshold':t,
                             'val_hit_rate':m['hit_rate'],'val_byte_hit_rate':m['byte_hit_rate'],'val_avg_latency_ms':m['avg_latency_ms'],'objective':objv(m)})

    # GATING (pilih base terbaik per-sampel di VAL, latih gate multi-class LR pada fitur = [proba base])
    if 'gating' in ensemble_types:
        keys=['gbdt','nb','elm','lr']; Pv=np.vstack([p_v[k] for k in keys]).T
        # indeks model terbaik w.r.t label: y=1 -> argmax p; y=0 -> argmin p
        best_idx=np.where(yv==1, np.argmax(Pv,axis=1), np.argmin(Pv,axis=1))
        gate=LogisticRegression(max_iter=1000, multi_class='auto').fit(Pv, best_idx)
        def gate_apply(P): 
            idx=gate.predict(P); rows=np.arange(len(P)); return P[rows, idx]
        pv=gate_apply(Pv)
        for a in size_alphas:
            for t in thresholds:
                m=eval_cache(df_v, pv, cfg, t, a)
                rows.append({'ensemble':'gating','weights':'{}','alpha':a,'threshold':t,
                             'val_hit_rate':m['hit_rate'],'val_byte_hit_rate':m['byte_hit_rate'],'val_avg_latency_ms':m['avg_latency_ms'],'objective':objv(m)})

    lb=pd.DataFrame(rows).sort_values('objective', ascending=False).reset_index(drop=True)
    lb_path=os.path.join(args.outdir,'tuning_leaderboard_val.csv'); lb.to_csv(lb_path, index=False)
    best=lb.iloc[0].to_dict()
    with open(os.path.join(args.outdir,'best_config.json'),'w') as f: json.dump(best,f,indent=2)
    print("[best@VAL]", best, flush=True)

    # evaluate best on TEST
    print("[test] evaluating best on TEST", flush=True)
    etype=best['ensemble']; a=float(best['alpha']); t=float(best['threshold'])
    if etype=='soft':
        w=json.loads(best['weights']); pte=soft_vote(p_te,w)
    elif etype=='hard':
        pte=hard_vote(p_te)
    elif etype=='stacking':
        Pte=np.vstack([p_te[k] for k in ['gbdt','nb','elm','lr']]).T
        pte=meta(Pte)
    else: # gating
        Pte=np.vstack([p_te[k] for k in ['gbdt','nb','elm','lr']]).T
        pte=gate_apply(Pte)
    mt=eval_cache(splits['test']['df_min'], pte, cfg, t, a)
    pd.DataFrame([mt]).to_csv(os.path.join(args.outdir,'tuning_test_best.csv'), index=False)
    print("[done] leaderboard:", lb_path, " test_best:", mt, flush=True)

if __name__=='__main__':
    main()