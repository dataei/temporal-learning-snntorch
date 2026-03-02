import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF

# Encoding: simple latency code (pixel -> spike time)
def latency_encode(x, num_steps=25, t_min=0, t_max=None):
    """
    x: [B, 1, 28, 28] in [0,1]
    returns spikes: [T, B, 1, 28, 28] {0,1}
    - brighter pixel => earlier spike
    - each pixel spikes at most once
    """
    if t_max is None:
        t_max = num_steps - 1

    x = x.clamp(0, 1)
    # map intensity to spike time (integer)
    t_spike = (t_max - (x * (t_max - t_min))).round().long()  # [B,1,28,28]

    # build spike train
    T = num_steps
    spikes = torch.zeros((T,) + x.shape, device=x.device)
    for t in range(T):
        spikes[t] = (t_spike == t).float()
    return spikes

# Model: 2 conv layers + FC, LIF after each layer
class ConvTemporalSNN(nn.Module):
    def __init__(self, beta=0.95, spike_grad="fast_sigmoid"):
        super().__init__()
        if spike_grad == "fast_sigmoid":
            sg = surrogate.fast_sigmoid(slope=25)
        elif spike_grad == "atan":
            sg = surrogate.atan()
        else:
            raise ValueError("Unknown surrogate")

        self.conv1 = nn.Conv2d(1, 12, 5, padding=2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)

        self.conv2 = nn.Conv2d(12, 32, 5, padding=2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)

        self.fc = nn.Linear(32 * 7 * 7, 10)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg)

        self.pool = nn.AvgPool2d(2)

    def forward(self, spike_in):
        """
        spike_in: [T, B, 1, 28, 28]
        returns:
          spk_out: [T, B, 10]
          mem_out: [T, B, 10]
          spike_counts: scalar-ish proxy
        """
        T, B = spike_in.shape[0], spike_in.shape[1]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        mem_out_rec = []
        spike_count = 0.0

        for t in range(T):
            x = spike_in[t] # [B,1,28,28]
            x = self.conv1(x)
            spk1, mem1 = self.lif1(x, mem1)
            spike_count += spk1.sum()

            x = self.pool(spk1) # [B,12,14,14]
            x = self.conv2(x)
            spk2, mem2 = self.lif2(x, mem2)
            spike_count += spk2.sum()

            x = self.pool(spk2) # [B,32,7,7]
            x = x.flatten(1)
            x = self.fc(x) # [B,10]
            spk_out, mem_out = self.lif_out(x, mem_out)
            spike_count += spk_out.sum()

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        spk_out_rec = torch.stack(spk_out_rec) # [T,B,10]
        mem_out_rec = torch.stack(mem_out_rec) # [T,B,10]
        return spk_out_rec, mem_out_rec, spike_count / B  # per-batch proxy

# Metrics: first spike time, time-to-decision, accuracy
def first_spike_time(spk_out):
    """
    spk_out: [T,B,C] binary spikes
    returns fst: [B,C] first spike timestep, or T if never spiked
    """
    T, B, C = spk_out.shape
    fst = torch.full((B, C), T, device=spk_out.device, dtype=torch.long)
    # find first time index where spk==1
    for t in range(T):
        fired = spk_out[t] > 0
        fst = torch.where((fst == T) & fired, torch.tensor(t, device=spk_out.device), fst)
    return fst

def predict_by_earliest_first_spike(spk_out):
    """
    chooses class with earliest first spike.
    if no class spikes, fallback to max membrane (handled outside if needed).
    """
    fst = first_spike_time(spk_out) # [B,C]
    pred = torch.argmin(fst, dim=1) # [B]
    ttd = torch.min(fst, dim=1).values # [B] time-to-decision
    return pred, ttd

# Train loop (temporal MSE loss)
def main(
    device="cuda" if torch.cuda.is_available() else "cpu",
    num_steps=25,
    batch_size=128,
    epochs=3,
    lr=1e-3,
    beta=0.95,
    T_on=5,
    T_off=None
):
    if T_off is None:
        T_off = num_steps - 1

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    net = ConvTemporalSNN(beta=beta).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # snnTorch temporal loss: penalize (t_actual - T_target)^2 for correct vs incorrect classes
    # expects target labels + desired on/off timing windows
    loss_fn = SF.mse_temporal_loss(on_target=T_on, off_target=T_off)

    for ep in range(1, epochs + 1):
        net.train()
        pbar = tqdm(train_dl, desc=f"epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            spk_in = latency_encode(x, num_steps=num_steps)

            spk_out, mem_out, spike_proxy = net(spk_in)

            loss = loss_fn(spk_out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred, ttd = predict_by_earliest_first_spike(spk_out.detach())
            acc = (pred == y).float().mean().item()
            pbar.set_postfix(loss=float(loss.item()), acc=acc, ttd=float(ttd.float().mean().item()), spikes=float(spike_proxy.item()))

        # quick eval
        net.eval()
        correct, total = 0, 0
        ttd_sum, spike_sum = 0.0, 0.0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                spk_in = latency_encode(x, num_steps=num_steps)
                spk_out, mem_out, spike_proxy = net(spk_in)

                pred, ttd = predict_by_earliest_first_spike(spk_out)
                correct += (pred == y).sum().item()
                total += y.numel()
                ttd_sum += ttd.float().sum().item()
                spike_sum += spike_proxy.item() * x.shape[0]

        print(f"[test] acc={correct/total:.4f}  mean_ttd={ttd_sum/total:.2f}  mean_spikes={spike_sum/total:.1f}")

if __name__ == "__main__":
    main()