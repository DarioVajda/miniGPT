# remove_bad_images.py
# Usage:
#   python remove_bad_images.py --bad_list bad_images.txt

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bad_list", required=True, help="Path to bad_images.txt file")
    ap.add_argument("--dry_run", action="store_true", help="Only print files that would be deleted")
    args = ap.parse_args()

    bad_list = Path(args.bad_list)
    if not bad_list.exists():
        raise FileNotFoundError(f"File not found: {bad_list}")

    with open(bad_list, "r", encoding="utf-8") as f:
        lines = f.readlines()

    deleted = 0
    for line in lines:
        # Each line format: "<path>\t<error>" from find_bad_images.py
        parts = line.strip().split("\t")
        if not parts:
            continue
        img_path = Path(parts[0])
        if img_path.exists():
            if args.dry_run:
                print(f"[DRY RUN] Would delete: {img_path}")
            else:
                try:
                    img_path.unlink()
                    print(f"Deleted: {img_path}")
                    deleted += 1
                except Exception as e:
                    print(f"Failed to delete {img_path}: {e}")
        else:
            print(f"File not found (already removed?): {img_path}")

    print(f"\nTotal deleted: {deleted}")

if __name__ == "__main__":
    main()