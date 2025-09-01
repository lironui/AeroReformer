import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# Define label classes and their corresponding RGB colors
CLASS_COLORS = {
    'Building': np.array([128, 0, 0]),
    'Road': np.array([128, 64, 128]),
    'Tree': np.array([0, 128, 0]),
    'Low_Vegetation': np.array([128, 128, 0]),
    'Moving_Car': np.array([64, 0, 128]),
    'Static_Car': np.array([192, 0, 192]),
    'Human': np.array([64, 64, 0]),
    'Clutter': np.array([0, 0, 0])
}

# Define minimum class presence thresholds (the fraction of pixels required)
CLASS_MIN_THRESHOLDS = {
    'Building': 0.15,
    'Tree': 0.2,
    'Road': 0.05,
    'Low_Vegetation': 0.1,
    'Moving_Car': 0.004,
    'Static_Car': 0.005,
    'Human': 0.001,
}

# Paths and parameters
PATCH_SIZE = 1024  # Original patch size (used for calculating distribution)
STRIDE = 1024
RESIZED_PATCH_SIZE = 1024  # Final saved patch size
DOWNSAMPLE_FACTOR = PATCH_SIZE // RESIZED_PATCH_SIZE


def extract_patches(image, label, patch_size, stride):
    """
    Extract 1024×1024 patches from the image and label.
    """
    patches = []
    h, w, _ = label.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[y:y + patch_size, x:x + patch_size]
            lbl_patch = label[y:y + patch_size, x:x + patch_size]
            patches.append((img_patch, lbl_patch, x, y))
    return patches


def filter_and_save_patches(patches, seq, img_name, output_dir):
    """
    Filter patches and save only the cropped RGB image directly to the output directory.
    """
    for i, (img_patch_orig, lbl_patch_orig, x, y) in enumerate(patches):
        # Calculate distribution on the original patch
        unique, counts = np.unique(lbl_patch_orig.reshape(-1, 3), axis=0, return_counts=True)
        total_pixels = lbl_patch_orig.shape[0] * lbl_patch_orig.shape[1]
        class_distribution = {
            tuple(color): count / total_pixels
            for color, count in zip(unique, counts)
        }

        # Identify the classes present (ignore 'Clutter')
        present_classes = [name for name, color in CLASS_COLORS.items() if tuple(color) in class_distribution]
        if "Clutter" in present_classes:
            present_classes.remove("Clutter")

        # Skip patch if only one class is present or if any class occupies more than 70%
        if len(present_classes) <= 1 or max(class_distribution.values()) > 0.7:
            continue

        # Downscale the image patch if necessary
        if PATCH_SIZE != RESIZED_PATCH_SIZE:
            img_patch_resized = cv2.resize(img_patch_orig, (RESIZED_PATCH_SIZE, RESIZED_PATCH_SIZE),
                                           interpolation=cv2.INTER_LINEAR)
        else:
            img_patch_resized = img_patch_orig

        # Convert from RGB (used in processing) to BGR (expected by cv2.imwrite)
        img_patch_bgr = cv2.cvtColor(img_patch_resized, cv2.COLOR_RGB2BGR)

        # Save one patch for each class that meets its threshold
        for cls in present_classes:
            if class_distribution.get(tuple(CLASS_COLORS[cls]), 0) < CLASS_MIN_THRESHOLDS.get(cls, 0.005):
                continue

            save_name = f"{seq}_{img_name}_{i}_{cls}.png"
            # Define the output path directly under the main output directory
            save_path_img = os.path.join(output_dir, save_name)

            # Save the cropped RGB image (as BGR)
            cv2.imwrite(save_path_img, img_patch_bgr)


def process_sequence(args):
    """
    Process all images within a sequence.
    """
    seq_path, seq, output_dir = args
    img_dir = os.path.join(seq_path, "Images")
    lbl_dir = os.path.join(seq_path, "Labels")

    for img_name in tqdm(sorted(os.listdir(img_dir)), desc=f"Processing {seq}"):
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name)

        if not os.path.isfile(lbl_path):
            continue

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(lbl_path, cv2.IMREAD_COLOR)
        if image is None or label is None:
            continue

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        patches = extract_patches(image, label, PATCH_SIZE, STRIDE)
        filter_and_save_patches(patches, seq, os.path.splitext(img_name)[0], output_dir)


def main():
    """
    Main function to parse arguments and start the processing pool.
    """
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Extract and filter patches from the UAVid dataset.")
    parser.add_argument(
        'folder',
        nargs='?',
        default='train',
        choices=['train', 'val', 'test'],
        help="Specify the dataset split to process: 'train', 'val', or 'test'. Defaults to 'train'."
    )
    args = parser.parse_args()
    folder = args.folder

    # 2. Define paths based on the parsed folder argument
    INPUT_DIR = f"data/UAVid_RIS/UAVid/uavid_{folder}"
    OUTPUT_DIR = f"data/UAVid_RIS/images/uavid_ris/PNGimages"

    print(f"✅ Processing folder: '{folder}'")
    print(f"➡️ Input directory:  {INPUT_DIR}")
    print(f"➡️ Output directory: {OUTPUT_DIR}")

    # 3. Create the main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sequences = [
        (os.path.join(INPUT_DIR, seq), seq, OUTPUT_DIR)
        for seq in sorted(os.listdir(INPUT_DIR))
        if os.path.isdir(os.path.join(INPUT_DIR, seq))
    ]

    if not sequences:
        print(f"⚠️ No sequences found in {INPUT_DIR}. Please check the path.")
        return

    with Pool(processes=16) as pool:
        pool.map(process_sequence, sequences)

    print("✅ Patch extraction complete.")


if __name__ == "__main__":
    main()