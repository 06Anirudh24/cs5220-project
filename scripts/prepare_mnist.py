import gzip
import struct
import numpy as np
from pathlib import Path


RAW_DIR = Path("/pscratch/sd/a/anirudh6/cs5220/project/data/mnist/raw")
OUT_DIR = Path("/pscratch/sd/a/anirudh6/cs5220/project/data/mnist/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_idx_images_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad image magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
        return data, rows, cols


def read_idx_labels_gz(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad label magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        if len(data) != num:
            raise ValueError(f"Label count mismatch in {path}: expected {num}, got {len(data)}")
        return data


def main():
    train_images, rows, cols = read_idx_images_gz(RAW_DIR / "train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels_gz(RAW_DIR / "train-labels-idx1-ubyte.gz")
    test_images, rows2, cols2 = read_idx_images_gz(RAW_DIR / "t10k-images-idx3-ubyte.gz")
    test_labels = read_idx_labels_gz(RAW_DIR / "t10k-labels-idx1-ubyte.gz")

    if (rows, cols) != (rows2, cols2):
        raise ValueError("Train/test image shape mismatch")

    # Normalize to [0,1] float32 for training
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Write flat binary files
    train_images.tofile(OUT_DIR / "train_images.bin")
    train_labels.tofile(OUT_DIR / "train_labels.bin")
    test_images.tofile(OUT_DIR / "test_images.bin")
    test_labels.tofile(OUT_DIR / "test_labels.bin")

    # Metadata
    with open(OUT_DIR / "metadata.txt", "w") as f:
        f.write(f"train_samples {train_images.shape[0]}\n")
        f.write(f"test_samples {test_images.shape[0]}\n")
        f.write(f"rows {rows}\n")
        f.write(f"cols {cols}\n")
        f.write(f"input_dim {rows * cols}\n")
        f.write("image_dtype float32\n")
        f.write("label_dtype uint8\n")
        f.write("layout row_major_flattened\n")
        f.write("normalization divide_by_255\n")

    print("Done.")
    print(f"Train images: {train_images.shape}, dtype={train_images.dtype}")
    print(f"Train labels: {train_labels.shape}, dtype={train_labels.dtype}")
    print(f"Test images:  {test_images.shape}, dtype={test_images.dtype}")
    print(f"Test labels:  {test_labels.shape}, dtype={test_labels.dtype}")
    print(f"Output written to: {OUT_DIR}")


if __name__ == "__main__":
    main()