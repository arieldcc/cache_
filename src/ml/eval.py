import argparse, os, yaml, json, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

def load_cfg(cfg_path):
    with open(cfg_path,'r') as f: return yaml.safe_load(f)

class LRUCache:
    def __init__(self, capacity_bytes):
        self.capacity=int(capacity_bytes); self.cur=0; self.store={}; self.order=[]
    def _touch(self,key):
        if key in self.store: self.order.remove(key); self.order.append(key)
    def get(self,key):
        hit = key in self.store
        if hit: self._touch(key)
        return hit
    def put(self,key,size):
        if key in self.store: self._touch(key); return
        while self.cur + size > self.capacity and self.order:
            k0=self.order.pop(0); s0,_=self.store.pop(k0); self.cur-=s0
        if size <= self.capacity:
            self.store[key]=(size,0); self.order.append(key); self.cur+=size

def eval_cache_policy(df_feat_test, df_lbl, proba, cfg, out_csv):
    # gabung label ke subset test pakai row_id agar alignment 1:1 dengan proba
    df = df_feat_test.merge(df_lbl[['row_id','label']], on='row_id', how='inner')
    df = df.sort_values('row_id').reset_index(drop=True)

    # pastikan panjang proba = panjang df
    if len(proba) != len(df):
        # jika tidak sama (harusnya tidak terjadi), selaraskan ke min panjang untuk menghindari index error
        m = min(len(proba), len(df))
        proba = proba[:m]
        df = df.iloc[:m].copy()

    thr=float(cfg['admission_threshold']); admit=(proba>=thr).astype(int)
    cache=LRUCache(cfg['capacity_bytes'])
    hits=reqs=bytes_hit=bytes_req=0; lat_sum=0.0; lat_hit=cfg['latency_ms']['hit']; lat_miss=cfg['latency_ms']['miss']

    for i, row in df.iterrows():
        size=int(row['size_bytes']); key=row['key']; reqs+=1; bytes_req+=size
        if cache.get(key):
            hits+=1; bytes_hit+=size; lat_sum+=lat_hit
        else:
            lat_sum+=lat_miss
            if admit[i]==1: cache.put(key,size)

    metrics={'requests':reqs,'hits':hits,'hit_rate':hits/max(reqs,1),
             'byte_hit_rate':bytes_hit/max(bytes_req,1),'avg_latency_ms':lat_sum/max(reqs,1),
             'total_bytes':bytes_req,'bytes_hit':bytes_hit}
    pd.DataFrame([metrics]).to_csv(out_csv, index=False); return metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--models_dir', required=True)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()

    cfg=load_cfg(args.cfg); os.makedirs(args.out, exist_ok=True)
    df_feat=pd.read_csv(args.features); df_lbl=pd.read_csv(args.labels)

    with open(os.path.join(args.models_dir,'feature_cols.json'),'r') as f:
        feature_cols=json.load(f)

    # bagi data seperti saat training
    n=len(df_feat); n_train=int(cfg['split']['train']*n); n_val=int(cfg['split']['val']*n); idx_test=slice(n_train+n_val, n)
    X=df_feat[feature_cols].values
    Xte=X[idx_test]
    df_feat_test = df_feat.iloc[idx_test].copy()

    ensemble=joblib.load(os.path.join(args.models_dir,'ensemble_soft.joblib'))
    lr_sgd=joblib.load(os.path.join(args.models_dir,'lr_sgd_decay.joblib'))

    p_ens=ensemble.predict_proba(Xte)[:,1]; p_lr=lr_sgd.predict_proba(Xte)[:,1]

    # Untuk metrik klasifikasi, kita butuh y_test; ambil dari label join by row_id dulu
    df_join = df_feat_test.merge(df_lbl[['row_id','label']], on='row_id', how='inner').sort_values('row_id')
    yte = df_join['label'].values
    # Selaraskan panjang jika ada selisih
    m = min(len(yte), len(p_ens), len(p_lr))
    yte = yte[:m]; p_ens = p_ens[:m]; p_lr = p_lr[:m]
    df_feat_test = df_join.iloc[:m].copy()  # untuk replay cache pakai df yang sudah align

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    def cls_metrics(p):
        yhat=(p>=0.5).astype(int)
        return {'accuracy':accuracy_score(yte,yhat),
                'precision':precision_score(yte,yhat,zero_division=0),
                'recall':recall_score(yte,yhat,zero_division=0),
                'f1':f1_score(yte,yhat,zero_division=0),
                'auc':roc_auc_score(yte,p)}

    ens_cls=cls_metrics(p_ens); lr_cls=cls_metrics(p_lr)
    pd.DataFrame([ens_cls]).to_csv(os.path.join(args.out,'classification_ensemble.csv'), index=False)
    pd.DataFrame([lr_cls]).to_csv(os.path.join(args.out,'classification_lr_sgd_decay.csv'), index=False)

    ens_cache=eval_cache_policy(df_feat_test, df_lbl, p_ens, cfg, os.path.join(args.out,'cache_ensemble.csv'))
    lr_cache=eval_cache_policy(df_feat_test, df_lbl, p_lr, cfg, os.path.join(args.out,'cache_lr_sgd_decay.csv'))

    yhat_ens=(p_ens>=0.5).astype(int); cm=confusion_matrix(yte,yhat_ens)
    import numpy as np; np.savetxt(os.path.join(args.out,'confusion_matrix_ensemble.csv'), cm.astype(int), fmt='%d', delimiter=',')

    summary={'ensemble_auc':ens_cls['auc'],'ensemble_f1':ens_cls['f1'],
             'ensemble_hit_rate':ens_cache['hit_rate'],'ensemble_byte_hit_rate':ens_cache['byte_hit_rate'],
             'ensemble_avg_latency_ms':ens_cache['avg_latency_ms'],
             'lr_auc':lr_cls['auc'],'lr_f1':lr_cls['f1'],
             'lr_hit_rate':lr_cache['hit_rate'],'lr_byte_hit_rate':lr_cache['byte_hit_rate'],
             'lr_avg_latency_ms':lr_cache['avg_latency_ms']}
    pd.DataFrame([summary]).to_csv(os.path.join(args.out,'summary.csv'), index=False)
    print('Saved results to', args.out)

if __name__ == '__main__':
    main()