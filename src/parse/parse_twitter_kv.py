import argparse, pandas as pd
from ..common.schema import TraceCols
def adapt(in_path,out_path):
    df = pd.read_csv(in_path); mapping={}
    if 'timestamp' in df.columns: mapping['timestamp']=TraceCols.ts
    if 'key' in df.columns: mapping['key']=TraceCols.key
    if 'value_size' in df.columns: mapping['value_size']=TraceCols.size
    if 'op' in df.columns: mapping['op']=TraceCols.op
    if 'ttl' in df.columns: mapping['ttl']=TraceCols.ttl
    df.rename(columns=mapping, inplace=True)
    if TraceCols.ttl not in df.columns: df[TraceCols.ttl] = -1.0
    df = df[[TraceCols.ts,TraceCols.key,TraceCols.size,TraceCols.op,TraceCols.ttl]].sort_values(TraceCols.ts)
    df.to_csv(out_path, index=False); print(f"Wrote adapted Twitter KV trace: {out_path} rows={len(df)}")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--in', dest='in_path', required=True); ap.add_argument('--out', dest='out_path', required=True)
    args = ap.parse_args(); adapt(args.in_path, args.out_path)
if __name__ == '__main__': main()
