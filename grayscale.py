import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_to_grayscale_3ch(img_path):
    """Convert RGB image to grayscale using luminance method, then back to 3-channel."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(gray_3ch, cv2.COLOR_RGB2BGR)  # back to BGR for saving

def process_dataset(src_dir, dst_dir):
    """Convert all images in a directory to 3-channel grayscale."""
    os.makedirs(dst_dir, exist_ok=True)
    for path in tqdm(sorted(Path(src_dir).rglob("*.[jpJP][pnPN]*[gG]")), desc=f"Processing {src_dir}"):
        rel_path = path.relative_to(src_dir)
        out_path = Path(dst_dir) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gray_img = convert_to_grayscale_3ch(path)
            cv2.imwrite(str(out_path), gray_img)
        except Exception as e:
            print(f"Skipping {path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert training/validation sets to grayscale for bias testing")
    parser.add_argument("--train_dir", required=True, help="Path to original RGB training set")
    parser.add_argument("--val_dir", required=True, help="Path to original RGB validation set")
    parser.add_argument("--test_dir", required=True, help="Path to original RGB test set")
    parser.add_argument("--out_root", required=True, help="Root output directory for grayscale sets")
    args = parser.parse_args()

    train_out = os.path.join(args.out_root, "grayscale_train")
    val_out = os.path.join(args.out_root, "grayscale_val")
    test_out = os.path.join(args.out_root, "test")  # untouched

    print("Converting training set...")
    process_dataset(args.train_dir, train_out)

    print("Converting validation set...")
    process_dataset(args.val_dir, val_out)

    print("Copying test set without modifications...")
    os.system(f"rsync -a {args.test_dir}/ {test_out}/")

    print(f"\n✅ Grayscale conversion completed. Output root: {args.out_root}")

if __name__ == "__main__":
    main()
