# dvsgesture_temporal.py
# DVSGesture experiments (MNIST-style): --Ton, --steps, --lam, SUMMARY
#
# Requirements:
#   pip install torch snntorch tonic tqdm numpy
#
# Run:
#   python dvsgesture_temporal.py --epochs 1 --steps 20 --Ton 6 --batch 2 --workers 0 --root ./data/DVSGesture

import argparse
import re
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
# Robust DVS128 .aedat reader (no read_dvs_ibm)
# ----------------------------
_EVENTS_DTYPE = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])

def read_events_dvs128_aedat(filename: str) -> np.ndarray:
    """
    Reads address-event data using tonic's low-level helpers and decodes address into x,y,p.
    Handles tonic versions where read_aedat_header_from_file returns 2 or 3 values.
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


def infer_label_from_name(aedat_path: str) -> int:
    """
    Infer class from 'gesture<number>' in path. Maps 1..11 -> 0..10.
    """
    s = Path(aedat_path).as_posix().lower()
    m = re.search(r"gesture(\d+)", s)
    if m:
        g = int(m.group(1))
        return max(0, min(10, g - 1))
    return 0


# ----------------------------
# Dataset
# ----------------------------
class DVSGestureRaw(Dataset):
    def __init__(self, root: str, train: bool, transform=None, H=64, W=64, strict_labels=False):
        self.root = Path(root)
        self.transform = transform
        self.H = H
        self.W = W
        self.strict_labels = strict_labels

        split_file = self.root / ("trials_to_train.txt" if train else "trials_to_test.txt")
        if not split_file.exists():
            raise FileNotFoundError(f"Missing {split_file}. root should be ./data/DVSGesture")

        with open(split_file, "r", encoding="utf-8") as f:
            trial_names = [ln.strip() for ln in f.readlines() if ln.strip()]

        aedat_files = list(self.root.rglob("*.aedat"))
        if len(aedat_files) == 0:
            raise RuntimeError(f"No .aedat files found under {self.root}")

        aedat_map = {p.stem: p for p in aedat_files}

        self.samples = []
        self.labels = []
        missing = []

        for name in trial_names:
            stem = Path(name).stem
            p = aedat_map.get(stem, None)
            if p is None:
                # fallback contains match
                matches = [pp for k, pp in aedat_map.items() if k.endswith(stem) or stem.endswith(k)]
                if matches:
                    p = matches[0]
                else:
                    missing.append(stem)
                    continue

            y = infer_label_from_name(str(p))
            if strict_labels and y == 0:
                raise RuntimeError(
                    f"Could not infer label from filename: {p}\n"
                    f"Your filenames may not contain gesture<number>."
                )

            self.samples.append(p)
            self.labels.append(y)

        if len(self.samples) == 0:
            raise RuntimeError("Resolved 0 samples. Check your split files vs .aedat filenames.")

        if missing:
            print(f"[warn] missing {len(missing)} trials (showing 5): {missing[:5]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        aedat_path = str(self.samples[idx])
        y = int(self.labels[idx])

        events = read_events_dvs128_aedat(aedat_path)

        # Manual downsample of coordinates BEFORE ToFrame to prevent out-of-bounds
        fx = 128 // self.W
        fy = 128 // self.H
        if fx <= 0 or fy <= 0 or (128 % self.W != 0) or (128 % self.H != 0):
            raise ValueError("H and W must divide 128 evenly (e.g., 64, 32).")

        events["x"] = (events["x"] // fx).astype(np.int16)
        events["y"] = (events["y"] // fy).astype(np.int16)

        # Clip just in case
        events["x"] = np.clip(events["x"], 0, self.W - 1)
        events["y"] = np.clip(events["y"], 0, self.H - 1)

        if self.transform is not None:
            frames = self.transform(events)
        else:
            frames = events

        return frames, y


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

        spk_out_rec = torch.stack(spk_out_rec)
        mem_out_rec = torch.stack(mem_out_rec)
        spike_proxy = spike_count / B
        return spk_out_rec, mem_out_rec, spike_proxy


# ----------------------------
# Temporal decoding
# ----------------------------
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


def collate_frames(batch):
    xs, ys = zip(*batch)
    xs = torch.stack([torch.as_tensor(x) for x in xs])
    ys = torch.tensor(ys, dtype=torch.long)

    if xs.ndim != 5:
        raise RuntimeError(f"Unexpected frame tensor shape: {xs.shape}")

    if xs.shape[-1] == 2:
        xs = xs.permute(0, 1, 4, 2, 3)  # [B,T,H,W,2] -> [B,T,2,H,W]
    elif xs.shape[2] == 2:
        pass
    else:
        raise RuntimeError(f"Unexpected channel layout: {xs.shape}")

    xs = (xs > 0).float()
    return xs, ys


# ----------------------------
# Train/Test
# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    T_off = min(args.steps - 1, args.Ton + 10)
    loss_fn = SF.mse_temporal_loss(on_target=args.Ton, off_target=T_off)

    # IMPORTANT: no TT.Downsample here — we already downsampled events in __getitem__
    transform = TT.Compose([
        TT.Denoise(filter_time=10000),
        TT.ToFrame(sensor_size=(args.H, args.W, 2), n_time_bins=args.steps),
    ])

    ds_train = DVSGestureRaw(args.root, train=True, transform=transform, H=args.H, W=args.W, strict_labels=args.strict_labels)
    ds_test  = DVSGestureRaw(args.root, train=False, transform=transform, H=args.H, W=args.W, strict_labels=args.strict_labels)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=False, collate_fn=collate_frames)
    dl_test  = DataLoader(ds_test, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=False, collate_fn=collate_frames)

    net = ConvTemporalSNN(num_classes=11, beta=args.beta).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        net.train()
        pbar = tqdm(dl_train, desc=f"epoch {ep}/{args.epochs}")

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
            for x, y in dl_test:
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
        print(f"SUMMARY dataset=DVSGestureRaw Ton={args.Ton} steps={args.steps} H={args.H} W={args.W} lam={args.lam}  "
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
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--strict_labels", action="store_true")
    args = p.parse_args()

    main(args)