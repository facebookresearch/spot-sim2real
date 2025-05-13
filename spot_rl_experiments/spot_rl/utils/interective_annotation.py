import json
import os
import pickle
import re
from glob import glob
from pathlib import Path

import cv2
import numpy as np


def normalize_folder_name(name):
    # Remove extensions and suffixes like _retry, _success, etc.
    return re.sub(r"(\.mp4)?(_[a-zA-Z_]+)?$", "", name)


def find_matching_folders(dir1, dir2):
    folders1 = {normalize_folder_name(Path(p).name): p for p in glob(f"{dir1}/*")}
    folders2 = {normalize_folder_name(Path(p).name): p for p in glob(f"{dir2}/*")}
    return {key: (folders1[key], folders2[key]) for key in folders1 if key in folders2}


def extract_frame_id(mask_filename):
    match = re.search(r"mask_(\d+)_", mask_filename)
    return int(match.group(1)) if match else None


def load_rgb_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)  # Expecting list of dicts with image info


def display_images(rgb_img, mask_img):
    mask_img = cv2.resize(mask_img, (rgb_img.shape[1], rgb_img.shape[0]))
    combined = np.hstack([rgb_img, mask_img])
    cv2.imshow("RGB (Left) + Mask (Right)", combined)


def prompt_user_input():
    print("Enter object name and furniture id (or [s] to skip, [q] to quit):")
    key = cv2.waitKey(0)
    if key == ord("s"):
        return "skip"
    elif key == ord("q"):
        return "quit"
    else:
        object_name = input("Object name: ").strip()
        furniture_id = input("Furniture ID: ").strip()
        return (object_name, furniture_id)


def main(dir1, dir2, output_json="annotations.json"):
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    folder_pairs = find_matching_folders(dir1, dir2)
    print(f"Found {len(folder_pairs)} matching folder pairs.")

    for base_name, (mask_dir, pkl_dir) in folder_pairs.items():
        print(f"\n📂 Processing: {base_name}")
        mask_files = sorted(glob(f"{mask_dir}/mask_*.png"))
        pkl_file = glob(f"{pkl_dir}/*.pkl")[0]
        rgb_data = load_rgb_data(pkl_file)

        if base_name not in annotations:
            annotations[base_name] = {}

        for mask_path in mask_files:
            frame_id = extract_frame_id(os.path.basename(mask_path))
            if frame_id is None or frame_id >= len(rgb_data):
                print(f"⚠️  Skipping invalid or out-of-range frame: {mask_path}")
                continue

            if str(frame_id) in annotations[base_name]:
                continue  # Already annotated

            rgb_img = rgb_data[frame_id].get("rgb")
            if rgb_img is None:
                print(f"⚠️  No RGB data for frame {frame_id}")
                continue

            mask_img = cv2.imread(mask_path)
            if mask_img is None or rgb_img is None:
                print(f"⚠️  Error loading images for frame {frame_id}")
                continue

            display_images(rgb_img, mask_img)
            response = prompt_user_input()

            if response == "skip":
                continue
            elif response == "quit":
                print("💾 Saving and exiting...")
                with open(output_json, "w") as f:
                    json.dump(annotations, f, indent=2)
                cv2.destroyAllWindows()
                return
            else:
                object_name, furniture_id = response
                annotations[base_name][str(frame_id)] = [object_name, furniture_id]
                print(f"✅ Saved: Frame {frame_id} → ({object_name}, {furniture_id})")

            # Periodically save
            with open(output_json, "w") as f:
                json.dump(annotations, f, indent=2)

    print("🎉 Done with all folders!")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", help="Directory with mask PNG folders")
    parser.add_argument("dir2", help="Directory with RGB pickle folders")
    parser.add_argument("--out", default="annotations.json", help="Output JSON file")
    args = parser.parse_args()
    main(args.dir1, args.dir2, args.out)
