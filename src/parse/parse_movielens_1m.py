import argparse, os, pandas as pd, numpy as np
from ..common.schema import TraceCols

def _read_dat(path, names):
    # Coba UTF-8, fallback ke latin-1 (MovieLens 1M sering butuh latin-1)
    try:
        return pd.read_csv(path, sep="::", engine="python", header=None, names=names, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep="::", engine="python", header=None, names=names, encoding="latin-1")

def load_ml1m(ml_dir):
    ratings = _read_dat(os.path.join(ml_dir, "ratings.dat"), ["user_id","movie_id","rating","timestamp"])
    users   = _read_dat(os.path.join(ml_dir, "users.dat"),   ["user_id","gender","age","occupation","zip"])
    movies  = _read_dat(os.path.join(ml_dir, "movies.dat"),  ["movie_id","title","genres"])
    df = ratings.merge(movies, on="movie_id", how="left").merge(users, on="user_id", how="left")
    return df

def compute_size_bytes(row, mode="constant", const_size_bytes=65536, base_size_kb=64):
    if mode == "constant":
        return int(const_size_bytes)
    gweights = {
        "Action": 8, "Adventure": 8, "Animation": 6, "Children's": 4, "Comedy": 6,
        "Crime": 6, "Documentary": 4, "Drama": 6, "Fantasy": 6, "Film-Noir": 4,
        "Horror": 6, "Musical": 5, "Mystery": 6, "Romance": 6, "Sci-Fi": 8,
        "Thriller": 7, "War": 7, "Western": 6
    }
    genres = str(row.get("genres") or "").split("|") if pd.notna(row.get("genres")) else []
    n = len(genres)
    kb = base_size_kb + 4*n + sum(gweights.get(g, 5) for g in genres)
    kb = max(kb, 1)
    return int(kb * 1024)

def adapt_ml1m(ml_dir, out_path, size_mode="constant", const_size_bytes=65536, base_size_kb=64):
    raw = load_ml1m(ml_dir)
    genres = raw["genres"].fillna("")
    primary = genres.apply(lambda s: s.split("|")[0] if s else "UNKNOWN")
    n_genres = genres.apply(lambda s: len(s.split("|")) if s else 0)

    out = pd.DataFrame({
        TraceCols.ts:   raw["timestamp"].astype(float),
        TraceCols.key:  raw["movie_id"].apply(lambda x: f"movie_{x}"),
        TraceCols.size: 0,
        TraceCols.op:   "GET",
        TraceCols.ttl:  -1.0,
        "user_id":      raw["user_id"].astype(int),
        "rating":       raw["rating"].astype(float),
        "gender":       raw["gender"].fillna("UNK"),
        "age":          raw["age"].fillna(-1).astype(int),
        "occupation":   raw["occupation"].fillna(-1).astype(int),
        "n_genres":     n_genres,
        "primary_genre":primary,
        "genres":       genres,
        "title":        raw["title"].fillna(""),
    })

    if size_mode == "constant":
        out[TraceCols.size] = int(const_size_bytes)
    else:
        out[TraceCols.size] = out.apply(lambda r: compute_size_bytes(
            r, mode="genre_weighted", const_size_bytes=const_size_bytes, base_size_kb=base_size_kb
        ), axis=1)

    out.sort_values(TraceCols.ts, inplace=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote MovieLens 1M unified trace: {out_path} rows={len(out)} "
          f"(size_mode={size_mode}, const={const_size_bytes}, base_kb={base_size_kb})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml_dir", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--size_mode", choices=["constant","genre_weighted"], default="genre_weighted")
    ap.add_argument("--const_size_bytes", type=int, default=65536)
    ap.add_argument("--base_size_kb", type=int, default=64)
    args = ap.parse_args()
    adapt_ml1m(args.ml_dir, args.out_path, args.size_mode, args.const_size_bytes, args.base_size_kb)

if __name__ == "__main__":
    main()