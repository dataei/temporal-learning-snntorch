# dvsgesture_temporal_segments.py
# DVSGesture (IBM) segment-based temporal-learning experiments (MNIST-style):
#   --Ton, --steps, --lam, SUMMARY
#
# Uses:
#   trials_to_train.txt / trials_to_test.txt  (recording list)
#   corresponding *_labels.csv                (segments with class + time range)
#
# Requirements:
#   pip install torch snntorch tonic tqdm numpy
#
# Run (smoke test):
#   python dvsgesture_temporal_segments.py --epochs 1 --steps 20 --Ton 6 --batch 2 --workers 0 --root ./data/DVSGesture

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

import tonic.transforms as TT
import tonic.io as tio


# ----------------------------
# Robust DVS128 .aedat reader (no tonic.read_dvs_ibm)
# ----------------------------
_EVENTS_DTYPE = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])

def read_events_dvs128_aedat(filename: str) -> np.ndarray:
    """
    Reads .aedat using tonic low-level helpers and decodes address into x,y,p.
    DVS128 address layout (common):
      bit 0    : polarity
      bits 1-7 : x
      bits 8-14: y
      bit 15   : special (ignore)
    """
    hdr = tio.read_aedat_header_from_file(filename)
    data_version = hdr[0]
    data_start = hdr[1]

    aer = tio.get_aer_events_from_file(filename, data_version, data_start)
    addr = aer["address"].astype(np.uint32)
    ts = aer["timeStamp"].astype(np.int64)

    # filter special events
    is_special = (addr & 0x8000) != 0
    addr = addr[~is_special]
    ts = ts[~is_special]

    p = (addr & 0x1).astype(np.int8)
    x = ((addr >> 1) & 0x7F).astype(np.int16)
    y = ((addr >> 8) & 0x7F).astype(np.int16)

    events = np.empty(x.shape[0], dtype=_EVENTS_DTYPE)
    events["x"] = x
    events["y"] = y
    events["t"] = ts
    events["p"] = p
    return events


def load_labels_csv(csv_path: Path):
    """
    Expects columns:
      class,startTime_usec,endTime_usec
    Returns list of (label0_10, t_start, t_end)
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "class,startTime_usec,endTime_usec":
            raise RuntimeError(f"Unexpected label CSV header in {csv_path}: {header}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            c, ts, te = line.split(",")
            c = int(c)              # 1..11
            y = max(0, min(10, c-1))  # -> 0..10
            t0 = int(ts)
            t1 = int(te)
            if t1 > t0:
                rows.append((y, t0, t1))
    return rows


# ----------------------------
# Segment dataset
# ----------------------------
class DVSGestureSegments(Dataset):
    """
    Each item is a labeled segment from a recording.
    """
    def __init__(self, root: str, train: bool, H: int, W: int, steps: int, cache_files: int = 2):
        self.root = Path(root)
        self.H = H
        self.W = W
        self.steps = steps

        if 128 % H != 0 or 128 % W != 0:
            raise ValueError("H and W must divide 128 evenly (e.g., 64 or 32).")

        split_file = self.root / ("trials_to_train.txt" if train else "trials_to_test.txt")
        if not split_file.exists():
            raise FileNotFoundError(f"Missing {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            rec_names = [ln.strip() for ln in f.readlines() if ln.strip()]

        # Build list of segments: (aedat_path, y, t0, t1)
        self.segments = []
        missing = []
        for rec in rec_names:
            aedat_path = self.root / rec
            labels_path = self.root / rec.replace(".aedat", "_labels.csv")
            if not aedat_path.exists():
                missing.append(str(aedat_path))
                continue
            if not labels_path.exists():
                missing.append(str(labels_path))
                continue

            segs = load_labels_csv(labels_path)
            for (y, t0, t1) in segs:
                self.segments.append((aedat_path, y, t0, t1))

        if missing:
            print("[warn] missing files (showing up to 5):")
            for m in missing[:5]:
                print("  ", m)

        if len(self.segments) == 0:
            raise RuntimeError("No segments found. Check that *_labels.csv files exist and are non-empty.")

        # Event -> frame transform (no Downsample here; we downsample coords ourselves)
        self.to_frame = TT.Compose([
            TT.Denoise(filter_time=10000),
            TT.ToFrame(sensor_size=(H, W, 2), n_time_bins=steps),
        ])

        # small LRU-ish cache for decoded full recordings
        self._cache_files = cache_files
        self._cache = {}      # path -> events
        self._cache_order = []  # paths in recency order

    def __len__(self):
        return len(self.segments)

    def _get_events_cached(self, aedat_path: Path) -> np.ndarray:
        key = str(aedat_path)
        if key in self._cache:
            # refresh recency
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]

        ev = read_events_dvs128_aedat(key)

        self._cache[key] = ev
        self._cache_order.append(key)
        # evict old
        while len(self._cache_order) > self._cache_files:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return ev

    def __getitem__(self, idx):
        aedat_path, y, t0, t1 = self.segments[idx]
        ev_full = self._get_events_cached(aedat_path)

        # slice time window
        m = (ev_full["t"] >= t0) & (ev_full["t"] < t1)
        ev = ev_full[m]
        if ev.size == 0:
            # return an empty-ish segment as zeros
            frames = np.zeros((self.steps, self.H, self.W, 2), dtype=np.uint8)
            return frames, y

        # shift time to start at 0 for this segment
        ev = ev.copy()
        ev["t"] = (ev["t"] - t0).astype(np.int64)

        # downsample coords to HxW
        fx = 128 // self.W
        fy = 128 // self.H
        ev["x"] = (ev["x"] // fx).astype(np.int16)
        ev["y"] = (ev["y"] // fy).astype(np.int16)
        ev["x"] = np.clip(ev["x"], 0, self.W - 1)
        ev["y"] = np.clip(ev["y"], 0, self.H - 1)

        frames = self.to_frame(ev)  # [T,H,W,2] or [T,2,H,W] depending on tonic version
        return frames, y


def collate_frames(batch):
    xs, ys = zip(*batch)
    xs = torch.stack([torch.as_tensor(x) for x in xs])  # [B, ...]
    ys = torch.tensor(ys, dtype=torch.long)

    if xs.ndim != 5:
        raise RuntimeError(f"Unexpected frame tensor shape: {xs.shape}")

    # [B,T,H,W,2] -> [B,T,2,H,W]
    if xs.shape[-1] == 2:
        xs = xs.permute(0, 1, 4, 2, 3)
    elif xs.shape[2] == 2:
        pass
    else:
        raise RuntimeError(f"Unexpected channel layout: {xs.shape}")

    xs = (xs > 0).float()
    return xs, ys


# ----------------------------
# Model
# ----------------------------
class ConvTemporalSNN(nn.Module):
    def __init__(self, num_classes=11, beta=0.95):
        super().__init__()
        sg = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(2, 12, 5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)

        self.conv2 = nn.Conv2d(12, 32, 5, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)

        self.pool = nn.AvgPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(32 * 7 * 7, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg)

    def forward(self, spike_in):
        T, B = spike_in.shape[0], spike_in.shape[1]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        mem_out_rec = []
        spike_count = 0.0

        for t in range(T):
            x = spike_in[t]
            x = self.conv1(x)
            spk1, mem1 = self.lif1(x, mem1)
            spike_count += spk1.sum()

            x = self.pool(spk1)
            x = self.conv2(x)
            spk2, mem2 = self.lif2(x, mem2)
            spike_count += spk2.sum()

            x = self.pool(spk2)
            x = self.gap(x)
            x = x.flatten(1)
            x = self.fc(x)

            spk_out, mem_out = self.lif_out(x, mem_out)
            spike_count += spk_out.sum()

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        spk_out_rec = torch.stack(spk_out_rec)  # [T,B,C]
        mem_out_rec = torch.stack(mem_out_rec)  # [T,B,C]
        spike_proxy = spike_count / B
        return spk_out_rec, mem_out_rec, spike_proxy


def first_spike_time(spk_out):
    T, B, C = spk_out.shape
    fst = torch.full((B, C), T, device=spk_out.device, dtype=torch.long)
    for t in range(T):
        fired = spk_out[t] > 0
        fst = torch.where((fst == T) & fired, torch.tensor(t, device=spk_out.device), fst)
    return fst

def predict_by_earliest_first_spike(spk_out):
    fst = first_spike_time(spk_out)
    pred = torch.argmin(fst, dim=1)
    ttd = torch.min(fst, dim=1).values
    return pred, ttd


# ----------------------------
# Train / Test
# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T_off = min(args.steps - 1, args.Ton + 10)
    loss_fn = SF.mse_temporal_loss(on_target=args.Ton, off_target=T_off)

    train_ds = DVSGestureSegments(args.root, train=True, H=args.H, W=args.W, steps=args.steps, cache_files=args.cache_files)
    test_ds  = DVSGestureSegments(args.root, train=False, H=args.H, W=args.W, steps=args.steps, cache_files=args.cache_files)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=False, collate_fn=collate_frames)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=False, collate_fn=collate_frames)

    net = ConvTemporalSNN(num_classes=11, beta=args.beta).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        net.train()
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{args.epochs}")

        for x, y in pbar:
            x = x.to(device)  # [B,T,2,H,W]
            y = y.to(device)

            spk_in = x.permute(1, 0, 2, 3, 4)  # [T,B,2,H,W]
            spk_out, mem_out, spike_proxy = net(spk_in)

            loss = loss_fn(spk_out, y)
            if args.lam != 0.0:
                loss = loss + args.lam * (spk_out.sum() / x.shape[0])

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred, ttd = predict_by_earliest_first_spike(spk_out.detach())
            acc = (pred == y).float().mean().item()
            pbar.set_postfix(loss=float(loss.item()),
                             acc=float(acc),
                             ttd=float(ttd.float().mean().item()),
                             spikes=float(spike_proxy.item()))

        # eval
        net.eval()
        correct, total = 0, 0
        ttd_sum, spike_sum = 0.0, 0.0
        with torch.no_grad():
            for x, y in test_dl:
                x = x.to(device)
                y = y.to(device)
                spk_in = x.permute(1, 0, 2, 3, 4)

                spk_out, mem_out, spike_proxy = net(spk_in)
                pred, ttd = predict_by_earliest_first_spike(spk_out)

                correct += (pred == y).sum().item()
                total += y.numel()
                ttd_sum += ttd.float().sum().item()
                spike_sum += spike_proxy.item() * x.shape[0]

        acc_test = correct / total
        mean_ttd = ttd_sum / total
        mean_spikes = spike_sum / total

        print(f"[test] acc={acc_test:.4f}  mean_ttd={mean_ttd:.2f}  mean_spikes={mean_spikes:.1f}")
        print(f"SUMMARY dataset=DVSGestureSegments Ton={args.Ton} steps={args.steps} H={args.H} W={args.W} lam={args.lam}  "
              f"acc={acc_test:.4f}  ttd={mean_ttd:.2f}  spikes={mean_spikes:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data/DVSGesture")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--Ton", type=int, default=6)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)      # keep 0 until stable
    p.add_argument("--cache_files", type=int, default=2)  # helps speed a lot
    args = p.parse_args()

    main(args)