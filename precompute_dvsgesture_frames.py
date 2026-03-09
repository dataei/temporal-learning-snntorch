import argparse
from pathlib import Path
import numpy as np
import tonic.transforms as TT
import tonic.io as tio
from tqdm import tqdm

_EVENTS_DTYPE = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])

def read_events_dvs128_aedat(filename: str) -> np.ndarray:
    hdr = tio.read_aedat_header_from_file(filename)
    data_version = hdr[0]
    data_start = hdr[1]
    aer = tio.get_aer_events_from_file(filename, data_version, data_start)
    addr = aer["address"].astype(np.uint32)
    ts = aer["timeStamp"].astype(np.int64)
    is_special = (addr & 0x8000) != 0
    addr = addr[~is_special]
    ts = ts[~is_special]
    p = (addr & 0x1).astype(np.int8)
    x = ((addr >> 1) & 0x7F).astype(np.int16)
    y = ((addr >> 8) & 0x7F).astype(np.int16)
    ev = np.empty(x.shape[0], dtype=_EVENTS_DTYPE)
    ev["x"], ev["y"], ev["t"], ev["p"] = x, y, ts, p
    return ev

def load_labels_csv(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "class,startTime_usec,endTime_usec":
            raise RuntimeError(f"Unexpected header in {csv_path}: {header}")
        for line in f:
            line=line.strip()
            if not line: continue
            c, t0, t1 = line.split(",")
            y = max(0, min(10, int(c)-1))
            t0 = int(t0); t1 = int(t1)
            if t1 > t0:
                rows.append((y, t0, t1))
    return rows

def main(args):
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split_file = root / ("trials_to_train.txt" if args.train else "trials_to_test.txt")
    recs = [ln.strip() for ln in split_file.read_text().splitlines() if ln.strip()]

    to_frame = TT.Compose([
        TT.Denoise(filter_time=10000),
        TT.ToFrame(sensor_size=(args.H, args.W, 2), n_time_bins=args.steps),
    ])

    fx = 128 // args.W
    fy = 128 // args.H

    seg_id = 0
    for rec in tqdm(recs, desc="recordings"):
        aedat_path = root / rec
        labels_path = root / rec.replace(".aedat", "_labels.csv")
        if not aedat_path.exists() or not labels_path.exists():
            continue

        ev_full = read_events_dvs128_aedat(str(aedat_path))
        segs = load_labels_csv(labels_path)

        for (y, t0, t1) in segs:
            m = (ev_full["t"] >= t0) & (ev_full["t"] < t1)
            ev = ev_full[m]
            if ev.size == 0:
                continue
            ev = ev.copy()
            ev["t"] = (ev["t"] - t0).astype(np.int64)
            ev["x"] = np.clip((ev["x"] // fx).astype(np.int16), 0, args.W-1)
            ev["y"] = np.clip((ev["y"] // fy).astype(np.int16), 0, args.H-1)

            frames = to_frame(ev).astype(np.uint8)  # [T,H,W,2] or [T,2,H,W]
            # normalize to [T,H,W,2]
            if frames.ndim == 4 and frames.shape[1] == 2:
                # [T,2,H,W] -> [T,H,W,2]
                frames = np.transpose(frames, (0, 2, 3, 1))

            np.savez_compressed(out / f"seg_{seg_id:07d}.npz", frames=frames, label=np.int64(y))
            seg_id += 1

    print("wrote", seg_id, "segments to", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data/DVSGesture")
    p.add_argument("--out", type=str, default="./data/DVSGesture/frames_cache")
    p.add_argument("--train", action="store_true")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=32)
    args = p.parse_args()
    main(args)