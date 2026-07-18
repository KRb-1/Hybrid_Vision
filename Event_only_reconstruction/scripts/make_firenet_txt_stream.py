import os
import sys
import numpy as np

WIDTH = 1280
HEIGHT = 720
CHUNK = 5_000_000

def process_one(npy_path, txt_path):
    arr = np.load(npy_path, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise RuntimeError(f"Bad npy shape: {arr.shape} for {npy_path}")

    total = arr.shape[0]
    print(f"\n==============================")
    print(f"NPY: {npy_path}")
    print(f"TXT: {txt_path}")
    print(f"shape: {arr.shape}")

    normal_valid = 0
    swapped_valid = 0

    # pass 1: decide x/y order
    for start in range(0, total, CHUNK):
        end = min(start + CHUNK, total)
        chunk = arr[start:end]

        col1 = chunk[:, 1]
        col2 = chunk[:, 2]

        normal_valid += np.count_nonzero(
            (col1 >= 0) & (col1 < WIDTH) & (col2 >= 0) & (col2 < HEIGHT)
        )
        swapped_valid += np.count_nonzero(
            (col2 >= 0) & (col2 < WIDTH) & (col1 >= 0) & (col1 < HEIGHT)
        )

    print("normal_valid:", int(normal_valid))
    print("swapped_valid:", int(swapped_valid))

    if swapped_valid > normal_valid:
        x_idx, y_idx = 2, 1
        print("Use swapped x/y")
    else:
        x_idx, y_idx = 1, 2
        print("Use normal x/y")

    # pass 2: write txt
    written = 0
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w") as f:
        f.write(f"{WIDTH} {HEIGHT}\n")

        for start in range(0, total, CHUNK):
            end = min(start + CHUNK, total)
            chunk = arr[start:end]

            t = chunk[:, 0]
            x = chunk[:, x_idx]
            y = chunk[:, y_idx]
            p = chunk[:, 3]

            valid = (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)

            if np.any(valid):
                out = np.column_stack([
                    t[valid],
                    x[valid].astype(np.int64),
                    y[valid].astype(np.int64),
                    p[valid].astype(np.int64)
                ])
                np.savetxt(f, out, fmt=["%.9f", "%d", "%d", "%d"])
                written += out.shape[0]

            print(f"processed {end}/{total}, written={written}", flush=True)

    print("saved:", txt_path)
    print("valid events written:", written)

def main():
    ids = sys.argv[1:]
    if not ids:
        raise ValueError("Please provide ids.")

    base = os.path.expanduser("~/autodl-tmp/HybridVision/memberC_results/converted_events")

    for sid in ids:
        npy_path = f"{base}/{sid}/events_txyp.npy"
        txt_path = f"{base}/{sid}/events_firenet.txt"
        process_one(npy_path, txt_path)

if __name__ == "__main__":
    main()
