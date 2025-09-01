import argparse, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    ap.add_argument('--decision_col', default='decision')
    args = ap.parse_args()
    df = pd.read_csv(args.in_path)
    if args.decision_col not in df.columns:
        raise ValueError(f"Column {args.decision_col} not found. Available: {list(df.columns)}")
    out = df[['ts','key', args.decision_col]].rename(columns={args.decision_col:'label'})
    import os
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    out.to_csv(args.out_path, index=False)
    print(f"Wrote labels from libCacheSim export: {args.out_path} rows={len(out)}")
if __name__ == '__main__':
    main()
