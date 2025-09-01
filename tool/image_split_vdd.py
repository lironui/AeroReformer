import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# VDD dataset class mappings (single-channel integer labels)
CLASS_LABELS = {
    0: "other",
    1: "wall",
    2: "road",
    3: "vegetation",
    4: "vehicles",
    5: "roof",
    6: "water"
}

# Minimum thresholds for class presence
CLASS_MIN_THRESHOLDS = {
    1: 0.08,  # wall
    2: 0.10,  # road
    3: 0.50,  # vegetation
    4: 0.005,  # vehicle
    5: 0.20,  # roof
    6: 0.05  # water
}

# Root path for VDD dataset
ROOT_DIR = "data/VDD_RIS/VDD"

# Output directory structure (flattened to a single folder)
OUTPUT_DIR = "data/VDD_RIS/images/vdd_ris/PNGimages"

# Patch extraction parameters
PATCH_SIZE = 1024
STRIDE = 1024
RESIZED_PATCH_SIZE = 1024
DOWNSAMPLE_FACTOR = PATCH_SIZE // RESIZED_PATCH_SIZE


def extract_patches(image_rgb, label_grey, patch_size, stride):
    """
    Extract 1024×1024 patches from an RGB image and single-channel label.
    """
    patches = []
    h, w = label_grey.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image_rgb[y:y + patch_size, x:x + patch_size]
            lbl_patch = label_grey[y:y + patch_size, x:x + patch_size]
            patches.append((img_patch, lbl_patch, x, y))
    return patches


def filter_and_save_patches(patches, split_name, img_base, out_dir):
    """
    Filter patches and save only the cropped RGB image directly to the output directory.
    """
    for i, (img_patch_orig, lbl_patch_orig, x, y) in enumerate(patches):
        total_pixels = lbl_patch_orig.size
        unique, counts = np.unique(lbl_patch_orig, return_counts=True)
        distribution = {int(u): (c / total_pixels) for u, c in zip(unique, counts)}

        # Exclude patches with only 1 class or any class >70%
        if len(distribution) <= 1 or max(distribution.values()) > 0.7:
            continue

        # Possibly downscale
        if PATCH_SIZE != RESIZED_PATCH_SIZE:
            img_patch_resized = cv2.resize(img_patch_orig, (RESIZED_PATCH_SIZE, RESIZED_PATCH_SIZE),
                                           interpolation=cv2.INTER_LINEAR)
        else:
            img_patch_resized = img_patch_orig

        # Convert from RGB -> BGR for saving
        img_patch_bgr = cv2.cvtColor(img_patch_resized, cv2.COLOR_RGB2BGR)

        # Process patches for each detected class that meets the threshold
        present_classes = [cid for cid in distribution.keys() if cid != 0]
        for cid in present_classes:
            frac = distribution[cid]
            if cid in CLASS_MIN_THRESHOLDS and frac < CLASS_MIN_THRESHOLDS[cid]:
                continue

            cls_name = CLASS_LABELS.get(cid, "unknown")
            patch_name = f"{split_name}_{img_base}_{i}_{cls_name}.png"

            # Define the output path directly under the main output directory
            out_img_path = os.path.join(out_dir, patch_name)

            # Save the cropped RGB image (as BGR)
            cv2.imwrite(out_img_path, img_patch_bgr)


def process_one_image(args):
    """
    Process a single image:
    - Read src image (JPG) and gt label (PNG)
    - Extract patches
    - Filter & Save patches
    """
    split_dir, split_name, img_base = args
    src_path = os.path.join(split_dir, "src", img_base + ".JPG")
    gt_path = os.path.join(split_dir, "gt", img_base + ".png")

    if not os.path.isfile(src_path) or not os.path.isfile(gt_path):
        return

    image_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    label_grey = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if label_grey is None:
        return

    # Extract & filter
    patches = extract_patches(image_rgb, label_grey, PATCH_SIZE, STRIDE)
    filter_and_save_patches(patches, split_name, img_base, OUTPUT_DIR)


def main():
    # 1. Set up argument parser
    parser = argparse.ArgumentParser(description="Extract patches from the VDD dataset.")
    parser.add_argument(
        'splits',
        nargs='*',  # 0 or more arguments
        choices=['train', 'val', 'test'],
        help="Specify dataset splits to process. Processes all if none are given."
    )
    args = parser.parse_args()

    # 2. Determine which splits to process
    splits_to_process = args.splits
    if not splits_to_process:
        splits_to_process = ['train', 'val', 'test']  # Default to all

    print(f"✅ Processing splits: {', '.join(splits_to_process)}")

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"➡️ Output will be saved to: {OUTPUT_DIR}")

    tasks = []
    # 3. Collect tasks from the specified splits
    for split in splits_to_process:
        split_dir = os.path.join(ROOT_DIR, split)
        src_dir = os.path.join(split_dir, "src")
        if not os.path.isdir(src_dir):
            print(f"⚠️ Warning: Directory not found for split '{split}': {src_dir}")
            continue

        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith(".jpg"):
                continue
            base_name = os.path.splitext(fname)[0]
            tasks.append((split_dir, split, base_name))

    if not tasks:
        print("No images found to process. Exiting.")
        return

    print(f"Found {len(tasks)} images total. Processing with 16 threads.")
    with Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(process_one_image, tasks), total=len(tasks)))

    print("✅ Patch extraction complete.")


if __name__ == "__main__":
    main()