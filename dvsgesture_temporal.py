import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

import tonic
import tonic.transforms as TT
from tonic.datasets import DVSGesture


# ----------------------------
# Model: Conv SNN (2-channel input), 11 classes
# ----------------------------
class ConvTemporalSNN(nn.Module):
    def __init__(self, num_classes=11, beta=0.95, spike_grad="fast_sigmoid"):
        super().__init__()
        if spike_grad == "fast_sigmoid":
            sg = surrogate.fast_sigmoid(slope=25)
        elif spike_grad == "atan":
            sg = surrogate.atan()
        else:
            raise ValueError("Unknown surrogate")

        # DVSGesture frames: 2 polarity channels
        self.conv1 = nn.Conv2d(2, 12, 5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)

        self.conv2 = nn.Conv2d(12, 32, 5, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)

        self.pool = nn.AvgPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Linear(32 * 7 * 7, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg)

    def forward(self, spike_in):
        """
        spike_in: [T, B, 2, H, W] (binary or float spikes/frames)
        returns:
          spk_out: [T, B, C]
          mem_out: [T, B, C]
          spike_proxy: scalar (mean spikes per sample across all layers)
        """
        T, B = spike_in.shape[0], spike_in.shape[1]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        mem_out_rec = []
        spike_count = 0.0

        for t in range(T):
            x = spike_in[t]             # [B,2,H,W]
            x = self.conv1(x)
            spk1, mem1 = self.lif1(x, mem1)
            spike_count += spk1.sum()

            x = self.pool(spk1)         # [B,12,H/2,W/2]
            x = self.conv2(x)
            spk2, mem2 = self.lif2(x, mem2)
            spike_count += spk2.sum()

            x = self.pool(spk2)         # [B,32, ...]
            x = self.gap(x)             # [B,32,7,7]
            x = x.flatten(1)
            x = self.fc(x)              # [B,C]
            spk_out, mem_out = self.lif_out(x, mem_out)
            spike_count += spk_out.sum()

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        spk_out_rec = torch.stack(spk_out_rec)  # [T,B,C]
        mem_out_rec = torch.stack(mem_out_rec)  # [T,B,C]
        spike_proxy = spike_count / B
        return spk_out_rec, mem_out_rec, spike_proxy


# ----------------------------
# Temporal metrics
# ----------------------------
def first_spike_time(spk_out):
    """
    spk_out: [T,B,C]
    returns fst: [B,C] first spike timestep, or T if never spiked
    """
    T, B, C = spk_out.shape
    fst = torch.full((B, C), T, device=spk_out.device, dtype=torch.long)
    for t in range(T):
        fired = spk_out[t] > 0
        fst = torch.where((fst == T) & fired, torch.tensor(t, device=spk_out.device), fst)
    return fst

def predict_by_earliest_first_spike(spk_out):
    fst = first_spike_time(spk_out)          # [B,C]
    pred = torch.argmin(fst, dim=1)          # [B]
    ttd = torch.min(fst, dim=1).values       # [B]
    return pred, ttd


# ----------------------------
# DVSGesture loader: events -> frames
# ----------------------------
def make_dvsgesture_loaders(data_root, batch_size, num_steps, H, W, num_workers=2):
    """
    Produces batches: x [B,T,2,H,W], y [B]
    """
    sensor_size = (128, 128, 2)

    # events -> frames with fixed time bins
    transform = TT.Compose([
        TT.Denoise(filter_time=10000),
        TT.Downsample(spatial_factor=128 // H),     # 128->H (must divide evenly)
        TT.ToFrame(sensor_size=(H, W, 2), n_time_bins=num_steps),
    ])

    # If tonic download is blocked (WAF), you must have data already present
    train_ds = DVSGesture(save_to=data_root, train=True, transform=transform)
    test_ds  = DVSGesture(save_to=data_root, train=False, transform=transform)

    def collate_frames(batch):
        xs, ys = zip(*batch)
        # xs: each element is frames; shape varies by tonic version
        xs = torch.stack([torch.as_tensor(x) for x in xs])  # [B, ...]
        ys = torch.tensor(ys, dtype=torch.long)

        # Expect either:
        # [B,T,H,W,2]  -> [B,T,2,H,W]
        # [B,T,2,H,W]  -> ok
        if xs.ndim != 5:
            raise RuntimeError(f"Unexpected DVSGesture frame tensor rank: {xs.shape}")

        if xs.shape[-1] == 2:
            xs = xs.permute(0, 1, 4, 2, 3)  # [B,T,H,W,2] -> [B,T,2,H,W]
        elif xs.shape[2] == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected DVSGesture channel placement: {xs.shape}")

        # binarize (simple baseline): any event count -> spike=1
        xs = (xs > 0).float()
        return xs, ys

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False,
                          collate_fn=collate_frames)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=False,
                         collate_fn=collate_frames)
    return train_dl, test_dl


# ----------------------------
# Train / Test
# ----------------------------
def main(
    data_root="./data",
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_steps=25,
    H=64,
    W=64,
    batch_size=16,
    epochs=3,
    lr=1e-3,
    beta=0.95,
    T_on=6,
    lam=0.0,
    num_workers=2
):
    # DVSGesture has 11 classes
    num_classes = 11

    # tighten off-target window (worked well for MNIST too)
    T_off = min(num_steps - 1, T_on + 10)

    train_dl, test_dl = make_dvsgesture_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_steps=num_steps,
        H=H, W=W,
        num_workers=num_workers
    )

    net = ConvTemporalSNN(num_classes=num_classes, beta=beta).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # temporal MSE loss: encourages correct class to spike near T_on, others near T_off
    loss_fn = SF.mse_temporal_loss(on_target=T_on, off_target=T_off)

    for ep in range(1, epochs + 1):
        net.train()
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{epochs}")

        for x, y in pbar:
            # x: [B,T,2,H,W] -> [T,B,2,H,W]
            x = x.to(device)
            y = y.to(device)
            spk_in = x.permute(1, 0, 2, 3, 4)

            spk_out, mem_out, spike_proxy = net(spk_in)

            loss = loss_fn(spk_out, y)
            # activity penalty (use output spikes only; same as MNIST path)
            if lam != 0.0:
                loss = loss + lam * (spk_out.sum() / x.shape[0])

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
        print(f"SUMMARY dataset=DVSGesture Ton={T_on} steps={num_steps} H={H} W={W} lam={lam}  acc={acc_test:.4f}  ttd={mean_ttd:.2f}  spikes={mean_spikes:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--Ton", type=int, default=6)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    main(
        data_root=args.data_root,
        num_steps=args.steps,
        H=args.H,
        W=args.W,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        T_on=args.Ton,
        lam=args.lam,
        num_workers=args.workers
    )