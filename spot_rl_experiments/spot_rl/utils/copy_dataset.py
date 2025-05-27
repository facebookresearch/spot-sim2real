import os
import re
import shutil
from datetime import datetime
from pathlib import Path

GOAT_LOG_PATTERN = re.compile(r"goat_log_(\d{4},\d{2},\d{2}-\d{2},\d{2},\d{2})")


def parse_timestamp_from_folder(name: str) -> datetime:
    """Extract datetime object from folder name like goat_log_2025,04,30-18,42,40"""
    match = GOAT_LOG_PATTERN.match(name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y,%m,%d-%H,%M,%S")


def get_valid_goat_log_subfolders(folder_path: Path):
    """Return list of (Path, size, timestamp) for valid goat_log_* subfolders."""
    subfolders = []
    for sub in folder_path.iterdir():
        if sub.is_dir() and GOAT_LOG_PATTERN.fullmatch(sub.name):
            timestamp = parse_timestamp_from_folder(sub.name)
            size = sum(f.stat().st_size for f in sub.glob("**/*") if f.is_file())
            subfolders.append((sub, size, timestamp))
    return subfolders


def select_largest_latest(subfolder_info):
    """Select the subfolder with largest size, and latest timestamp if tie."""
    if not subfolder_info:
        return None
    # Sort by size (desc), then timestamp (desc)
    subfolder_info.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return subfolder_info[0][0]  # Return Path


def copy_largest_data_pkl(source_root: Path):
    source_root = Path(source_root).resolve()
    if not source_root.exists():
        print(f"❌ Error: source directory does not exist: {source_root}")
        return

    # Dynamically compute destination
    dest_root = source_root.parent / "fremont_dataset"
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output will be saved to: {dest_root}")

    for folder in source_root.iterdir():
        if not folder.is_dir():
            continue

        normalized_name = folder.name
        valid_logs = get_valid_goat_log_subfolders(folder)

        if not valid_logs:
            print(
                f"⚠️  Skipping {normalized_name} (no valid goat_log_* subfolders found)"
            )
            continue

        best_subfolder = select_largest_latest(valid_logs)
        data_pkl = best_subfolder / "data.pkl"
        if not data_pkl.exists():
            print(f"⚠️  No data.pkl found in {best_subfolder}")
            continue

        dest_folder = dest_root / normalized_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest_path = dest_folder / "data.pkl"

        shutil.copy2(data_pkl, dest_path)
        print(f"✅ Copied: {data_pkl} → {dest_path}")

    print("🎉 Done copying all largest-latest data.pkl files.")


# === CLI Entry ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Copy largest-latest data.pkl from goat_log folders to fremont_dataset."
    )
    parser.add_argument(
        "source_dir", help="Root directory with instructionX_variant_Y folders"
    )
    args = parser.parse_args()

    copy_largest_data_pkl(args.source_dir)
