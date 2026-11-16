#!/usr/bin/env python
"""Extract AIGVDBench and write its manifest (the paper's training / validation / test protocol).

Download first:  huggingface-cli download AIGVDBench/AIGVDBench --repo-type dataset --local-dir data/AIGVDBench
Protocol (official split files Split/{train,val,test}.jsonl):
  train = Open-Sora (T2V) 14,000 + real 14,000 (same Video_id), val 3,000 + 3,000, test 3,000 + 3,000;
  the other 19 open-source generators: their 3,000 test clips each; 11 closed-source generators: 2,000 clips each.
Output: <out>/<group>/<subset>/<id>.mp4 and <out>/manifest_aigvd.csv with
  dataset, subset, group, split, label, video_id, pair_id, path.
Re-running skips files that already exist; --n-* < 0 means "all".

    python scripts/prepare_aigvdbench.py --root data/AIGVDBench --out data/aigvd
"""
import argparse, csv, json, os, random, time, zipfile

OPEN = {
    "OpenSource/T2V/Open-Sora.zip": ("open_t2v", "OpenSora-T2V"),
    "OpenSource/T2V/AccVideo.zip": ("open_t2v", "AccVideo-T2V"),
    "OpenSource/T2V/AnimateDiff.zip": ("open_t2v", "AnimateDiff-T2V"),
    "OpenSource/T2V/Cogvideox1.5.zip": ("open_t2v", "CogVideoX1.5-T2V"),
    "OpenSource/T2V/EasyAnimate.zip": ("open_t2v", "EasyAnimate-T2V"),
    "OpenSource/T2V/HunyuanVideo.zip": ("open_t2v", "HunyuanVideo-T2V"),
    "OpenSource/T2V/IPOC.zip": ("open_t2v", "IPOC-T2V"),
    "OpenSource/T2V/LTX.zip": ("open_t2v", "LTX-T2V"),
    "OpenSource/T2V/Pyramid-Flow.zip": ("open_t2v", "PyramidFlow-T2V"),
    "OpenSource/T2V/RepVideo.zip": ("open_t2v", "RepVideo-T2V"),
    "OpenSource/T2V/VideoCrafter.zip": ("open_t2v", "VideoCrafter-T2V"),
    "OpenSource/T2V/Wan2.1.zip": ("open_t2v", "Wan2.1-T2V"),
    "OpenSource/I2V/EasyAnimate.zip": ("open_i2v", "EasyAnimate-I2V"),
    "OpenSource/I2V/LTX.zip": ("open_i2v", "LTX-I2V"),
    "OpenSource/I2V/Pyramid-Flow.zip": ("open_i2v", "PyramidFlow-I2V"),
    "OpenSource/I2V/SEINE.zip": ("open_i2v", "SEINE-I2V"),
    "OpenSource/I2V/SVD.zip": ("open_i2v", "SVD-I2V"),
    "OpenSource/I2V/VideoCrafter.zip": ("open_i2v", "VideoCrafter-I2V"),
    "OpenSource/V2V/Cogvideox1.5.zip": ("open_v2v", "CogVideoX1.5-V2V"),
    "OpenSource/V2V/LTX.zip": ("open_v2v", "LTX-V2V"),
}
CLOSED = {f"ClosedSource/{z}.zip": ("closed", n) for z, n in [
    ("causvid_24fps", "CausVid"), ("Gen2", "Gen2"), ("Gen3", "Gen3"), ("Jimeng", "Jimeng"), ("kling", "Kling"),
    ("Luma", "Luma"), ("Opensora", "OpenSora-web"), ("pika", "Pika"), ("Sora", "Sora"), ("vidu", "Vidu"), ("wan", "Wan")]}


def read_ids(p):
    return [json.loads(l)["Video_id"] for l in open(p) if l.strip()]


def extract(zpath, members, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    ok = miss = 0
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        by_base = {os.path.basename(n): n for n in names if n.lower().endswith(".mp4")}
        for m in members:
            if m not in names:
                m = by_base.get(os.path.basename(m))
                if m is None:
                    miss += 1
                    continue
            dst = os.path.join(dst_dir, os.path.basename(m))
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                ok += 1
                continue
            with z.open(m) as src, open(dst + ".part", "wb") as f:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            os.replace(dst + ".part", dst)
            ok += 1
    return ok, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="downloaded dataset folder (contains AIGVDBench/ and Split/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=-1)
    ap.add_argument("--n-val", type=int, default=-1)
    ap.add_argument("--n-test", type=int, default=-1, help="test clips per non-Open-Sora generator (-1 = all)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    Z, S = os.path.join(a.root, "AIGVDBench"), os.path.join(a.root, "Split")
    rng = random.Random(a.seed)
    pick = lambda ids, n: sorted(ids) if n < 0 or n >= len(ids) else sorted(rng.sample(ids, n))
    train, val, test = read_ids(f"{S}/train.jsonl"), read_ids(f"{S}/val.jsonl"), read_ids(f"{S}/test.jsonl")
    train_sel, val_sel, test_sel = pick(train, a.n_train), pick(val, a.n_val), pick(test, a.n_test)
    closed_idx = list(range(2000)) if a.n_test < 0 else sorted(rng.sample(range(2000), min(a.n_test, 2000)))
    rows, t0 = [], time.time()

    def add(subset, group, split, label, ids, d):
        for i in ids:
            vid = i[:-4] if i.endswith(".mp4") else i
            rows.append(dict(dataset="aigvd", subset=subset, group=group, split=split, label=label,
                             video_id=f"{subset}/{vid}", pair_id=vid, path=os.path.join(d, os.path.basename(i))))

    d = os.path.join(a.out, "real", "Real")
    for split, ids in (("train", train_sel), ("val", val_sel), ("test", test)):
        ok, miss = extract(f"{Z}/Real/Real.zip", [f"videos/{i}" for i in ids], d)
        add("Real", "real", split, 0, ids, d)
        print(f"Real/{split}: {ok} ok, {miss} missing, {time.time() - t0:.0f}s", flush=True)
    for zrel, (group, subset) in OPEN.items():
        d = os.path.join(a.out, group, subset)
        plan = (("train", train_sel), ("val", val_sel), ("test", test)) if subset == "OpenSora-T2V" else (("test", test_sel),)
        for split, ids in plan:
            ok, miss = extract(f"{Z}/{zrel}", [f"videos/{i}" for i in ids], d)
            add(subset, group, split, 1, ids, d)
            print(f"{subset}/{split}: {ok} ok, {miss} missing, {time.time() - t0:.0f}s", flush=True)
    for zrel, (group, subset) in CLOSED.items():
        d = os.path.join(a.out, group, subset)
        with zipfile.ZipFile(f"{Z}/{zrel}") as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".mp4"))
        members = [names[i] for i in closed_idx if i < len(names)]
        ok, miss = extract(f"{Z}/{zrel}", members, d)
        add(subset, group, "test", 1, [os.path.basename(m) for m in members], d)
        print(f"{subset}/test: {ok} ok, {miss} missing, {time.time() - t0:.0f}s", flush=True)
    mp = os.path.join(a.out, "manifest_aigvd.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {mp}")


if __name__ == "__main__":
    main()
