import argparse, pandas as pd, numpy as np
from ..common.schema import TraceCols
from collections import defaultdict, deque

def label_relaxed_belady(df: pd.DataFrame, lookahead=5000) -> pd.DataFrame:
    df = df.copy()
    df['row_id'] = df.index.astype(int)
    df = df.reset_index(drop=True)

    n = len(df)
    next_pos = defaultdict(deque)
    for i in range(n-1, -1, -1):
        next_pos[df.at[i, TraceCols.key]].appendleft(i)

    size_thresh = np.percentile(df[TraceCols.size], 95)
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        k = df.at[i, TraceCols.key]
        if next_pos[k] and next_pos[k][0] == i:
            next_pos[k].popleft()
        if next_pos[k]:
            nxt = next_pos[k][0]
            if nxt - i <= lookahead and df.at[i, TraceCols.size] <= size_thresh:
                labels[i] = 1

    out = pd.DataFrame({
        'row_id': df['row_id'],
        'ts': df[TraceCols.ts],
        'key': df[TraceCols.key],
        'label': labels
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    ap.add_argument('--capacity_bytes', type=int, default=128*1024*1024)  # unused in this relaxed proxy
    ap.add_argument('--lookahead', type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    labels = label_relaxed_belady(df, lookahead=args.lookahead)
    labels.to_csv(args.out_path, index=False)
    print(f"Wrote relaxed Belady labels: {args.out_path} rows={len(labels)}")

if __name__ == '__main__':
    main()