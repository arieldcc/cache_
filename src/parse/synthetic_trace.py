import argparse, numpy as np, pandas as pd
from ..common.schema import TraceCols
def synth_sequence(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    n_keys = max(1000, int(0.05*n))
    keys = [f"obj_{i}" for i in range(n_keys)]
    key_probs = np.array([1.0/(i+1) for i in range(n_keys)]); key_probs /= key_probs.sum()
    ts = 0.0; rows = []
    for i in range(n):
        k = rng.choice(keys, p=key_probs)
        size = int(rng.choice([512,1024,4096,16384,65536,262144], p=[.1,.2,.3,.2,.15,.05]))
        ttl = float(rng.choice([30,60,300,1800,-1], p=[.2,.2,.2,.1,.3]))
        ts += float(rng.gamma(1.0, 0.05))
        rows.append((ts,k,size,"GET",ttl))
        if rng.random() < 0.02: rows.append((ts,k,size,"PUT",ttl))
    df = pd.DataFrame(rows, columns=[TraceCols.ts,TraceCols.key,TraceCols.size,TraceCols.op,TraceCols.ttl])
    return df
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); df = synth_sequence(args.n, args.seed); df.to_csv(args.out, index=False); print(f"Wrote synthetic trace: {args.out} rows={len(df)}")
if __name__ == "__main__": main()
