import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

def spikes_to_frames(times, units, T=25, W=700):
    frames = np.zeros((T, 1, 1, W), dtype=np.float32)

    if len(times) == 0:
        return frames

    tmax = times.max()
    if tmax <= 0:
        return frames

    bins = np.linspace(0, tmax, T + 1)

    for t, u in zip(times, units):
        idx = np.searchsorted(bins, t, side="right") - 1
        if 0 <= idx < T and 0 <= u < W:
            frames[idx, 0, 0, u] += 1.0

    mx = frames.max()
    if mx > 0:
        frames /= mx

    return frames

def process_file(h5_file, out_dir, T):
    with h5py.File(h5_file, "r") as f:
        times = f["spikes/times"]
        units = f["spikes/units"]
        labels = f["labels"]

        for i in tqdm(range(len(labels)), desc=f"processing {Path(h5_file).name}"):
            t = times[i]
            u = units[i]
            y = int(labels[i])

            frames = spikes_to_frames(t, u, T=T, W=700)

            np.savez_compressed(
                out_dir / f"{i:06d}.npz",
                frames=frames,
                label=y
            )

def main():
    T = 25

    train_h5 = "data/ssc/ssc_train.h5"
    test_h5  = "data/ssc/ssc_test.h5"

    train_out = Path("data/ssc/cache_T25_train")
    test_out  = Path("data/ssc/cache_T25_test")

    train_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    process_file(train_h5, train_out, T)
    process_file(test_h5, test_out, T)

if __name__ == "__main__":
    main()