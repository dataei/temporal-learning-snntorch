# dvsgesture_train_from_cache.py
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF


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
    xs = torch.stack([torch.as_tensor(x) for x in xs])
    ys = torch.tensor(ys, dtype=torch.long)

    if xs.ndim != 5:
        raise RuntimeError(f"Unexpected cached frame shape: {xs.shape}")

    # DVSGesture cache: [B,T,H,W,2] -> [B,T,2,H,W]
    if xs.shape[-1] == 2:
        xs = xs.permute(0, 1, 4, 2, 3)

    # SHD cache: [B,T,1,1,700] already means [B,T,C,H,W]
    elif xs.shape[2] == 1:
        pass

    else:
        raise RuntimeError(
            f"Unexpected cached frame shape: {xs.shape} "
            f"(expected DVSGesture [B,T,H,W,2] or SHD [B,T,1,1,700])"
        )

    xs = xs.float()

    # normalize per sample
    mx = xs.amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1.0)
    xs = xs / mx

    return xs, ys


class ConvTemporalSNN(nn.Module):
    def __init__(self, num_classes=20, beta=0.95):
        super().__init__()
        sg = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, 12, kernel_size=(1, 5), padding=(0, 2))
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)

        self.conv2 = nn.Conv2d(12, 32, kernel_size=(1, 5), padding=(0, 2))
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)

        self.pool = nn.AvgPool2d((1, 2))
        self.gap = nn.AdaptiveAvgPool2d((1, 32))

        self.fc = nn.Linear(32 * 1 * 32, num_classes)
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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = CachedFrames(args.train_cache)
    test_ds = CachedFrames(args.test_cache)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=False, collate_fn=collate_cached)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=False, collate_fn=collate_cached)

    net = ConvTemporalSNN(num_classes=20, beta=args.beta).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # temporal loss (only used in temporal mode)
    T_off = min(args.steps - 1, args.Ton + 10)
    temporal_loss = SF.mse_temporal_loss(on_target=args.Ton, off_target=T_off)

    for ep in range(1, args.epochs + 1):
        net.train()
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)  # [B,T,2,H,W]
            y = y.to(device)

            spk_in = x.permute(1, 0, 2, 3, 4)  # [T,B,2,H,W]
            spk_out, mem_out, spike_proxy = net(spk_in)

            # ---- define logits/loss per mode (ALWAYS sets loss) ----
            if args.mode == "temporal":
                loss = temporal_loss(spk_out, y)
                pred, ttd = predict_by_earliest_first_spike(spk_out.detach())
            elif args.mode == "rate_mem":
                logits = mem_out[-1]                 # [B,C]
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(1)
                ttd = torch.full((y.shape[0],), args.steps, device=device, dtype=torch.long)
            elif args.mode == "rate_spk":
                logits = spk_out.sum(0)              # [B,C]
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(1)
                ttd = torch.full((y.shape[0],), args.steps, device=device, dtype=torch.long)
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # optional spike penalty
            if args.lam != 0.0:
                loss = loss + args.lam * (spk_out.sum() / x.shape[0])

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = (pred == y).float().mean().item()
            pbar.set_postfix(loss=float(loss.item()),
                             acc=float(acc),
                             ttd=float(ttd.float().mean().item()),
                             spikes=float(spike_proxy.item()))

        # ---- eval ----
        net.eval()
        correct, total = 0, 0
        ttd_sum, spike_sum = 0.0, 0.0

        with torch.no_grad():
            for x, y in test_dl:
                x = x.to(device)
                y = y.to(device)
                spk_in = x.permute(1, 0, 2, 3, 4)
                spk_out, mem_out, spike_proxy = net(spk_in)

                if args.mode == "temporal":
                    pred, ttd = predict_by_earliest_first_spike(spk_out)
                elif args.mode == "rate_mem":
                    logits = mem_out[-1]
                    pred = logits.argmax(1)
                    ttd = torch.full((y.shape[0],), args.steps, device=device, dtype=torch.long)
                else:  # rate_spk
                    logits = spk_out.sum(0)
                    pred = logits.argmax(1)
                    ttd = torch.full((y.shape[0],), args.steps, device=device, dtype=torch.long)

                correct += (pred == y).sum().item()
                total += y.numel()
                ttd_sum += ttd.float().sum().item()
                spike_sum += spike_proxy.item() * x.shape[0]

        acc_test = correct / total
        mean_ttd = ttd_sum / total
        mean_spikes = spike_sum / total

        print(f"[test] acc={acc_test:.4f}  mean_ttd={mean_ttd:.2f}  mean_spikes={mean_spikes:.1f}")
        print(f"SUMMARY mode={args.mode} dataset=SHDCached Ton={args.Ton} steps={args.steps} H={args.H} W={args.W} lam={args.lam}  "
              f"acc={acc_test:.4f}  ttd={mean_ttd:.2f}  spikes={mean_spikes:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_cache", type=str, required=True)
    p.add_argument("--test_cache", type=str, required=True)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--Ton", type=int, default=6)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--mode", type=str, default="temporal", choices=["temporal", "rate_mem", "rate_spk"])
    args = p.parse_args()
    main(args)