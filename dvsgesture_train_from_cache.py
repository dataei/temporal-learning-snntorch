# dvsgesture_train_from_cache.py
# Train DVSGesture experiments from precomputed frame cache (FAST).
#
# Cache format: each file is .npz with:
#   frames: uint8 array shape [T,H,W,2]  (event counts per bin, polarity channel last)
#   label : int (0..10)
#
# Requirements:
#   pip install torch snntorch tqdm numpy
#
# Example:
#   python dvsgesture_train_from_cache.py \
#     --train_cache ./data/DVSGesture/cache_T10_H32_W32_train \
#     --test_cache  ./data/DVSGesture/cache_T10_H32_W32_test  \
#     --epochs 3 --steps 10 --Ton 3 --H 32 --W 32 --batch 64 --workers 2
#
# Then do sweeps like MNIST:
#   --Ton 3/6/10 and --lam 0 vs 3e-6

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


# ----------------------------
# Dataset: cached .npz frames
# ----------------------------
class CachedFrames(Dataset):
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache dir not found: {self.cache_dir}")

        self.files = sorted(self.cache_dir.glob("*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {self.cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        d = np.load(fp, allow_pickle=False)
        frames = d["frames"]  # [T,H,W,2] uint8
        label = int(d["label"])
        return frames, label


def collate_cached(batch):
    xs, ys = zip(*batch)
    xs = torch.stack([torch.as_tensor(x) for x in xs])  # [B,T,H,W,2]
    ys = torch.tensor(ys, dtype=torch.long)

    if xs.ndim != 5 or xs.shape[-1] != 2:
        raise RuntimeError(f"Unexpected cached frame shape: {xs.shape} (expected [B,T,H,W,2])")

    # [B,T,H,W,2] -> [B,T,2,H,W]
    xs = xs.permute(0, 1, 4, 2, 3)

    # binarize to spikes
    xs = (xs > 0).float()
    return xs, ys


# ----------------------------
# Model (same as before)
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
        # spike_in: [T,B,2,H,W]
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
        mem_out_rec = torch.stack(mem_out_rec)
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

    train_ds = CachedFrames(args.train_cache)
    test_ds = CachedFrames(args.test_cache)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=False, collate_fn=collate_cached
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=False, collate_fn=collate_cached
    )

    net = ConvTemporalSNN(num_classes=11, beta=args.beta).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Tight off-target window
    T_off = min(args.steps - 1, args.Ton + 10)
    loss_fn = SF.mse_temporal_loss(on_target=args.Ton, off_target=T_off)

    for ep in range(1, args.epochs + 1):
        net.train()
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{args.epochs}")
        for x, y in pbar:
            # x: [B,T,2,H,W] -> [T,B,2,H,W]
            x = x.to(device)
            y = y.to(device)

            spk_in = x.permute(1, 0, 2, 3, 4)
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
        print(f"SUMMARY dataset=DVSGestureCached Ton={args.Ton} steps={args.steps} H={args.H} W={args.W} lam={args.lam}  "
              f"acc={acc_test:.4f}  ttd={mean_ttd:.2f}  spikes={mean_spikes:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", type=str, required=True)
    p.add_argument("--test_cache", type=str, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--Ton", type=int, default=3)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    main(args)