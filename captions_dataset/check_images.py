# find_bad_images.py
# Usage:
#   python find_bad_images.py --root ./captions_dataset/train --out bad_images.txt
#   (extensions default: jpg,jpeg,png,webp)

from __future__ import annotations
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import sys

from PIL import Image, UnidentifiedImageError

def check_image(path: Path) -> tuple[Path | None, str | None]:
    """
    Returns (None, None) if image is OK, else (path, error_string).
    We verify, then reopen and convert to RGB to mimic training.
    """
    try:
        with Image.open(path) as im:
            im.verify()  # quick structural check
        with Image.open(path) as im:
            _ = im.convert("RGB")  # force decode like your pipeline
        return (None, None)
    except Exception as e:
        return (path, f"{type(e).__name__}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory with images (scans recursively).")
    ap.add_argument("--exts", default="jpg,jpeg,png,webp", help="Comma-separated extensions to scan.")
    ap.add_argument("--out", default=None, help="Optional path to write bad file list.")
    ap.add_argument("--workers", type=int, default=8, help="Thread workers (IO-bound).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    exts = {e.lower().lstrip(".") for e in args.exts.split(",")}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in exts]
    print(f"Scanning {len(files):,} files under {root} …")

    bad: list[tuple[Path, str]] = []
    err_types = Counter()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(check_image, p): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            bad_path, err = fut.result()
            if bad_path is not None:
                bad.append((bad_path, err))
                err_types[err.split(":")[0]] += 1
            if i % 1000 == 0:
                print(f"  …checked {i:,}/{len(files):,} (bad so far: {len(bad):,})")

    # Print results
    if bad:
        print("\nBad images:")
        for p, err in bad:
            print(f"{p}  ||  {err}")
    else:
        print("\nNo problematic images found 🎉")

    print("\nSummary:")
    print(f"  Total files: {len(files):,}")
    print(f"  Bad files  : {len(bad):,}")
    if bad:
        print("  Error types:")
        for k, v in err_types.most_common():
            print(f"    {k}: {v}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            for p, err in bad:
                f.write(f"{p}\t{err}\n")
        print(f"\nWrote list to {outp}")

if __name__ == "__main__":
    main()