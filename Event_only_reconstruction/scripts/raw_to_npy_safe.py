import os
import sys
import numpy as np
from numpy.lib.format import open_memmap
from metavision_core.event_io import EventsIterator

if len(sys.argv) != 3:
    print("Usage: python3.10 raw_to_npy_safe.py <input_raw> <output_npy>")
    sys.exit(1)

raw_path = sys.argv[1]
out_path = sys.argv[2]
tmp_path = out_path + ".tmp.npy"

if not os.path.exists(raw_path):
    raise FileNotFoundError(raw_path)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

print("Input raw:", raw_path)
print("Output npy:", out_path)

total = 0
for evs in EventsIterator(input_path=raw_path, delta_t=100000):
    total += len(evs)

print("Total events:", total)

if total <= 0:
    raise RuntimeError("No events read from raw file.")

if os.path.exists(tmp_path):
    os.remove(tmp_path)

arr = open_memmap(tmp_path, mode="w+", dtype=np.float64, shape=(total, 4))

pos = 0
for evs in EventsIterator(input_path=raw_path, delta_t=100000):
    n = len(evs)
    if n == 0:
        continue

    arr[pos:pos+n, 0] = evs["t"].astype(np.float64)
    arr[pos:pos+n, 1] = evs["x"].astype(np.float64)
    arr[pos:pos+n, 2] = evs["y"].astype(np.float64)
    arr[pos:pos+n, 3] = evs["p"].astype(np.float64)

    pos += n

arr.flush()
del arr

check = np.load(tmp_path, mmap_mode="r")
assert check.ndim == 2 and check.shape[1] == 4
assert check.shape[0] == total

print("Check OK:", check.shape)
print("t min/max:", float(check[:, 0].min()), float(check[:, 0].max()))
print("x min/max:", float(check[:, 1].min()), float(check[:, 1].max()))
print("y min/max:", float(check[:, 2].min()), float(check[:, 2].max()))
print("p unique:", np.unique(check[:, 3]))

os.replace(tmp_path, out_path)
print("Saved final:", out_path)
