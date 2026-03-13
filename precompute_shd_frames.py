import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def spikes_to_frames(times, units, T=25, W=700):
    frames = np.zeros((T, 1, 1, W), dtype=np.float32)

    tmax = times.max()
    bins = np.linspace(0, tmax, T + 1)

    for t, u in zip(times, units):
        idx = np.searchsorted(bins, t) - 1
        if 0 <= idx < T:
            frames[idx, 0, 0, u] += 1

    frames /= frames.max() + 1e-6
    return frames

def process_file(h5_file, out_dir, T):

    with h5py.File(h5_file, "r") as f:

        times = f["spikes/times"]
        units = f["spikes/units"]
        labels = f["labels"]

        for i in tqdm(range(len(labels))):

            t = times[i]
            u = units[i]

            frames = spikes_to_frames(t, u, T=T)

            label = int(labels[i])

            np.savez(
                out_dir / f"{i:06d}.npz",
                frames=frames,
                label=label
            )


def main():

    T = 25

    train_h5 = "data/SHD/shd_train.h5"
    test_h5  = "data/SHD/shd_test.h5"

    train_out = Path("data/SHD/cache_T25_train")
    test_out = Path("data/SHD/cache_T25_test")

    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    process_file(train_h5, train_out, T)
    process_file(test_h5, test_out, T)


if __name__ == "__main__":
    main()