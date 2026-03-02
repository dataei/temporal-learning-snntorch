import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from tqdm import tqdm

# ===========================
# 1. Advanced Metrics Logic
# ===========================
def first_spike_time(spk_out):
    """Returns the first spike timestamp [B, C] for each class."""
    T, B, C = spk_out.shape
    fst = torch.full((B, C), float(T), device=spk_out.device)
    for t in range(T):
        fired = spk_out[t] > 0
        fst = torch.where((fst == T) & fired, torch.tensor(float(t), device=spk_out.device), fst)
    return fst

def predict_by_ttd(spk_out):
    """Winner-Take-All: Earliest spike wins the classification."""
    fst = first_spike_time(spk_out)
    ttd, pred = torch.min(fst, dim=1) # ttd = Time to Decision
    return pred, ttd

# ===========================
# 2. Merged Architecture (64-filter User Architecture)
# ===========================
class MergedTemporalSNN(nn.Module):
    def __init__(self, beta=0.9, slope=25):
        super().__init__()
        self.beta = beta
        spike_grad = surrogate.fast_sigmoid(slope=slope)

        # Layer 1: 12 filters
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        # Layer 2: 64 filters (User's high-accuracy config)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        # Output
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spike_in):
        T, B = spike_in.shape[0], spike_in.shape[1]
        mem1, mem2, mem_out = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif_out.init_leaky()
        
        spk_out_rec = []
        total_spikes = 0

        for t in range(T):
            x = spike_in[t]
            
            # Layer 1 dynamics
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            x = self.pool1(spk1)
            total_spikes += spk1.sum()

            # Layer 2 dynamics
            cur2 = self.conv2(x)
            spk2, mem2 = self.lif2(cur2, mem2)
            x = self.pool2(spk2)
            total_spikes += spk2.sum()

            # Output dynamics
            x = self.flatten(x)
            cur_out = self.fc1(x)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            total_spikes += spk_out.sum()

            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec), total_spikes / B

# ===========================
# 3. Main Training Experiment
# ===========================
def run_experiment(T_on=5, num_steps=25, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment: T_on={T_on} | Steps={num_steps}")

    # Data Loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    net = MergedTemporalSNN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = SF.mse_temporal_loss(on_target=T_on, off_target=num_steps-1)

    for epoch in range(epochs):
        net.train()
        loop = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            # Latency encoding outside forward pass for efficiency
            from snntorch import spikegen
            spk_in = spikegen.latency(x, num_steps=num_steps, threshold=0.01)

            spk_out, avg_spikes = net(spk_in)
            loss = loss_fn(spk_out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track Metrics
            pred, ttd = predict_by_ttd(spk_out.detach())
            acc = (pred == y).float().mean()
            loop.set_postfix(acc=f"{acc:.2%}", ttd=f"{ttd.mean():.1f}", spikes=int(avg_spikes))

if __name__ == "__main__":
    run_experiment(T_on=5) # Start with fast target