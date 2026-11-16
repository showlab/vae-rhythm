#!/usr/bin/env python
"""DF26 (test only) -> manifest. Download (gated; accept the terms on the dataset page first):
    huggingface-cli download DF26/DF26 --repo-type dataset --local-dir data/DF26
Columns: dataset=df26, subset=generator ("Real" for real clips), group (real / open_t2v / open_i2v / closed),
label, video_id (relative path without extension), pair_id (clip id of the paired real video -> paired 2AFC), path.

    python scripts/prepare_df26.py --root data/DF26 --out data/df26/manifest_df26.csv
"""
import argparse, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = pd.read_csv(os.path.join(a.root, "metadata", "videos.csv"))
    real_clip = dict(zip(v[v.label == "real"].video_path, v[v.label == "real"].clip_id))
    rows = []
    for r in v.itertuples():
        fake = r.label == "fake"
        acc = str(r.generator_access) if fake else ""
        rows.append(dict(
            dataset="df26", subset=r.generator if fake else "Real",
            group=("closed" if acc == "closed" else ("open_i2v" if str(r.generator_variant) == "i2v" else "open_t2v")) if fake else "real",
            label=int(fake), video_id=os.path.splitext(r.video_path)[0], path=os.path.join(a.root, r.video_path),
            pair_id=real_clip.get(r.paired_real_path, "") if fake else r.clip_id, split="test", scenario=r.scenario))
    m = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    m.to_csv(a.out, index=False)
    print(f"{len(m)} rows ({(m.label == 0).sum()} real / {(m.label == 1).sum()} generated) -> {a.out}")
    print(m.groupby(["group", "subset"]).size().to_string())


if __name__ == "__main__":
    main()
