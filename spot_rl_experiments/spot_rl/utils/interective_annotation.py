import json
import os
import pickle
import re
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

MAPPING = {
    "bed": 1,
    "white_chair": 13,
    "white_chair_in_back": 14,
    "coffee_table": 18,
    "sofa": 19,
    "dining_table": 20,
    "sink": 23,
    "living_room_console": 24,
    "tv_console": 25,
    "dresser": 26,
    "kitchen_island": 28,
    "kitchen_counter": 29,
    "left_nightstand": 30,
    "right_nightstand": 31,
    "office_desk": 32,
    "ign": 100,
}
REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}


def find_matching_folders(mask_root, rgb_root):
    """
    Matches folders by stripping only the `.mp4` extension from mask_root filenames
    and checking if the resulting name exists in rgb_root.
    """
    mask_root = Path(mask_root)
    rgb_root = Path(rgb_root)

    masks = {}
    rgbs = {}

    for pf in mask_root.iterdir():
        if pf.is_dir() and pf.suffix == ".mp4":
            key = pf.stem  # e.g., instruction2_variant_C_retry_partial_failure
            masks[key] = str(pf)

    for p in rgb_root.iterdir():
        if p.is_dir():
            key = p.name  # Keep full folder name
            rgbs[key] = str(p)

    common_keys = set(masks) & set(rgbs)
    unmatched_masks = set(masks) - common_keys
    unmatched_rgbs = set(rgbs) - common_keys
    print(f"🔗 Found {len(common_keys)} matching folder pairs.")
    print(f"❌ Unmatched in mask_root (.mp4): {len(unmatched_masks)}")
    for k in sorted(unmatched_masks):
        print(f"  - {k}")
    print(f"❌ Unmatched in rgb_root (folders): {len(unmatched_rgbs)}")
    for k in sorted(unmatched_rgbs):
        print(f"  - {k}")
    return {key: (masks[key], rgbs[key]) for key in sorted(common_keys)}


def extract_frame_id(mask_filename):
    match = re.search(r"mask_(\d+)_", mask_filename)
    return int(match.group(1)) if match else None


def extract_object_name(mask_filename):
    """
    Extracts object name from a filename like:
    - mask_003_donut.png         → 'donut'
    - mask_013_plush_toy_1.png   → 'plush_toy'
    - mask_042_dining_chair_3.png → 'dining_chair'
    """
    match = re.search(r"mask_\d+_(.+)\.png$", mask_filename)
    if not match:
        return None

    object_with_suffix = match.group(1)
    # Remove trailing _number suffix, if present
    clean_name = re.sub(r"_\d+$", "", object_with_suffix)
    return clean_name


def load_rgb_data(pkl_path):
    with open(pkl_path, "rb") as f:
        full_data = pickle.load(f)  # list of dicts, each with "camera_source"

    # Extract hand_color_image per frame
    filtered = {}
    for i, frame in enumerate(full_data):
        sources = frame.get("camera_data", [])
        hand_image_data = next(
            (
                d["raw_image"]
                for d in sources
                if d.get("src_info") == "hand_color_image"
            ),
            None,
        )
        if hand_image_data is not None:
            filtered[i] = hand_image_data
        else:
            print(f"⚠️ No hand_color_image found in frame {i}")
            filtered[i] = None  # or skip / raise if needed

    return filtered


# Global figure objects so the window persists
_fig, _ax = None, None


def display_images(rgb_img, mask_img):
    global _fig, _ax

    mask_resized = cv2.resize(mask_img, (rgb_img.shape[1], rgb_img.shape[0]))
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2RGB)

    combined = np.hstack([rgb_bgr, mask_bgr])

    if _fig is None:
        plt.ion()
        _fig, _ax = plt.subplots(figsize=(12, 6))
        _ax.axis("off")
        _im = _ax.imshow(combined)
        plt.title("Left: RGB  |  Right: Mask")
        plt.pause(0.001)
    else:
        _ax.clear()
        _ax.axis("off")
        _ax.imshow(combined)
        plt.draw()
        plt.pause(0.001)


def prompt_user_input(mode, current_annotation=None, display_object_name=None):
    if mode == "annotate":
        key = input("Enter furniture name (or 'skip'): ").strip().lower()
        if key == "skip":
            return "skip"
        furniture_id = MAPPING.get(key, None)
        if furniture_id is None:
            print(f"Input '{key}' does not exist in mappings. Try again.")
            return prompt_user_input(mode)
        return furniture_id

    elif mode == "verify":
        print(f"Current annotation: {current_annotation}")
        response = input("Confirm annotation? (y/n/skip): ").strip().lower()
        if response == "y":
            return "confirm"
        elif response == "n":
            return prompt_user_input("annotate")
        elif response == "skip":
            return "skip"
        else:
            print("Invalid input. Please enter 'y', 'n', or 'skip'.")
            return prompt_user_input(mode, current_annotation, display_object_name)


def main(dir1, dir2, output_json="annotations.json", mode="annotate"):
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    folder_pairs = find_matching_folders(dir1, dir2)
    print(f"Found {len(folder_pairs)} matching folder pairs.")

    def sort_key(name):
        match = re.match(r"instruction(\d+[a-z]*)_.*variant_([A-Z])", name)
        if not match:
            return (9999, "Z")  # fallback for unexpected formats
        instr_num = match.group(1)
        variant = match.group(2)
        # Convert '3a' to something sortable — e.g., int('3') = 3 and 'a' = small offset
        digits = re.match(r"(\d+)([a-z]?)", instr_num)
        instr_base = int(digits.group(1))
        suffix = digits.group(2)
        instr_index = instr_base * 10 + (ord(suffix) - ord("a") + 1 if suffix else 0)
        return (instr_index, variant)

    # Sorted list of (key, (mask_dir, pkl_dir)) tuples
    sorted_folder_pairs = sorted(
        folder_pairs.items(), key=lambda item: sort_key(item[0])
    )

    skipped_frames = {}

    for base_name, (mask_dir, pkl_dir) in sorted_folder_pairs:
        print(f"\n📂 Processing: {base_name}")

        # Hack skip until we fix this.
        if base_name == "instruction1_variant_C":
            continue

        mask_files = sorted(glob(f"{mask_dir}/mask_*.png"))
        pkl_file = glob(f"{pkl_dir}/*.pkl")[0]
        rgb_data = load_rgb_data(pkl_file)

        if base_name not in annotations:
            annotations[base_name] = {}

        for mask_path in mask_files:
            frame_id = extract_frame_id(os.path.basename(mask_path))
            display_object_name = extract_object_name(os.path.basename(mask_path))
            if frame_id is None or frame_id >= len(rgb_data.keys()):
                print(f"⚠️  Skipping invalid or out-of-range frame: {mask_path}")
                continue

            rgb_img = rgb_data[frame_id]
            if rgb_img is None:
                print(f"⚠️  No RGB data for frame {frame_id}")
                continue

            mask_img = cv2.imread(mask_path)
            if mask_img is None:
                print(f"⚠️  Error loading mask image for frame {frame_id}")
                continue

            print(f"🖼️  Frame : {frame_id} , Object - {display_object_name}")
            display_images(rgb_img, mask_img)

            frame_key = str(frame_id)
            if base_name not in annotations:
                annotations[base_name] = {}

            if args.mode == "verify" and frame_key in annotations[base_name]:
                existing_all = annotations[base_name][frame_key]
                existing = [
                    item for item in existing_all if item[0] == display_object_name
                ]

                if not existing:
                    add_new = input(
                        f"⚠️ No existing annotation found for object '{display_object_name}' in frame {frame_id}. Add it? (y/n)"
                    )
                    if add_new == "y":
                        if frame_key not in annotations[base_name]:
                            annotations[base_name][frame_key] = []

                        response = prompt_user_input("annotate")
                        if response == "skip":
                            skipped_frames.setdefault(base_name, []).append(frame_id)
                            continue
                        annotations[base_name][frame_key].append(
                            (display_object_name, response)
                        )
                        print(
                            f"✅ Saved: Frame {frame_id} → ({display_object_name}, {response})"
                        )
                    else:
                        continue
                else:
                    # There could still be multiple entries with the same object name – usually one, but we handle all
                    existing_readable = [
                        (obj, REVERSE_MAPPING.get(fid, f"unknown({fid})"))
                        for obj, fid in existing
                    ]
                    result = prompt_user_input(
                        "verify",
                        current_annotation=existing_readable,
                        display_object_name=display_object_name,
                    )

                    if result == "confirm":
                        continue  # keep existing annotation
                    elif result == "skip":
                        skipped_frames.setdefault(base_name, []).append(frame_id)
                        continue  # do not modify existing
                    else:
                        annotations[base_name][frame_key] = [
                            (display_object_name, result)
                        ]
                        print(
                            f"🔁 Updated: Frame {frame_id} → ({display_object_name}, {result})"
                        )
            elif args.mode == "annotate":
                if frame_key not in annotations[base_name]:
                    annotations[base_name][frame_key] = []

                response = prompt_user_input("annotate")
                if response == "skip":
                    skipped_frames.setdefault(base_name, []).append(frame_id)
                    continue
                annotations[base_name][frame_key].append(
                    (display_object_name, response)
                )
                print(
                    f"✅ Saved: Frame {frame_id} → ({display_object_name}, {response})"
                )

            # # Save after each annotation
            with open(output_json, "w") as f:
                json.dump(annotations, f, indent=2)

    print("🎉 Done with all folders!")

    if skipped_frames:
        print("\n📋 Skipped Frames Summary:")
        for k, v in skipped_frames.items():
            print(f"  {k} → {sorted(v)}")
    else:
        print("✅ No frames were skipped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--masks_dir", help="Directory with mask PNG folders")
    parser.add_argument("--log_dir", help="Directory with RGB pickle folders")
    parser.add_argument("--out", default="annotations.json", help="Output JSON file")
    parser.add_argument("--mode", default="annotate", help="annotate or verify")
    args = parser.parse_args()
    main(args.masks_dir, args.log_dir, args.out, args.mode)
