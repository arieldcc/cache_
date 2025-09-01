import argparse, pandas as pd
from ..common.schema import TraceCols
def adapt(in_path, out_path):
    df = pd.read_csv(in_path)
    colmap = {'ts':TraceCols.ts, 'key':TraceCols.key, 'size':TraceCols.size}
    for c0,c1 in colmap.items():
        if c0 in df.columns and c0!=c1: df.rename(columns={c0:c1}, inplace=True)
    if TraceCols.op not in df.columns: df[TraceCols.op]='GET'
    if TraceCols.ttl not in df.columns: df[TraceCols.ttl]=-1.0
    df = df[[TraceCols.ts,TraceCols.key,TraceCols.size,TraceCols.op,TraceCols.ttl]].sort_values(TraceCols.ts)
    df.to_csv(out_path, index=False); print(f"Wrote adapted LRB/Wiki trace: {out_path} rows={len(df)}")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--in', dest='in_path', required=True); ap.add_argument('--out', dest='out_path', required=True)
    args = ap.parse_args(); adapt(args.in_path, args.out_path)
if __name__ == '__main__': main()
