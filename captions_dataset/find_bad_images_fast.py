# find_bad_images_fast.py
# Usage:
#   python find_bad_images_fast.py --root ./captions_dataset/train --out bad_images.txt
#   (extensions default: jpg,jpeg,png,webp)

from __future__ import annotations
import argparse, os, sys, time
from typing import Iterable, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter

from PIL import Image, ImageFile

# Be strict: truncated images should raise instead of silently loading
ImageFile.LOAD_TRUNCATED_IMAGES = False

def iter_files(root: str, exts: set[str]) -> Iterable[str]:
    """Stream file paths under root matching extensions (case-insensitive)."""
    exts = {e.lower().lstrip(".") for e in exts}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # Fast, case-insensitive ext check without Path() allocation
            dot = fn.rfind(".")
            if dot <= 0: 
                continue
            if fn[dot+1:].lower() in exts:
                yield os.path.join(dirpath, fn)

def check_image(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (None, None) if image is OK, else (path, error_string).
    Decode fully, then convert to RGB to mimic training.
    """
    # Quick zero-size check avoids PIL open attempt
    try:
        if os.path.getsize(path) == 0:
            return path, "EmptyFile: file size is 0 bytes"
    except OSError as e:
        return path, f"OSError: {e}"

    try:
        with Image.open(path) as im:
            # Ensure full decode
            im.load()
            # Mimic training decode/convert
            _ = im.convert("RGB")
        return None, None
    except Exception as e:
        return path, f"{type(e).__name__}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory with images (scans recursively).")
    ap.add_argument("--exts", default="jpg,jpeg,png,webp", help="Comma-separated extensions to scan.")
    ap.add_argument("--out", default=None, help="Optional path to write bad file list (written incrementally).")
    ap.add_argument("--workers", type=int, default=0, help="Concurrency. 0 => auto (min(64, cpu*4)).")
    ap.add_argument("--prefetch", type=int, default=4, help="Pending batches per worker (bounds memory).")
    ap.add_argument("--mp", action="store_true", help="Use ProcessPoolExecutor instead of threads.")
    ap.add_argument("--progress-every", type=int, default=5000, help="Print progress every N files checked.")
    args = ap.parse_args()

    root = args.root
    if not os.path.exists(root):
        print(f"Root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    exts = set(args.exts.split(","))
    cpu = os.cpu_count() or 8
    workers = args.workers or min(64, cpu * 4)  # IO-heavy default
    prefetch = max(1, args.prefetch)
    progress_every = max(1, args.progress_every)

    t0 = time.time()
    print(f"Scanning under {root} (extensions: {','.join(sorted({e.lower() for e in exts}) )}) …")
    print(f"Executor: {'Process' if args.mp else 'Thread'}Pool, workers={workers}, prefetch={prefetch}")

    bad_count = 0
    total_checked = 0
    err_types = Counter()

    # Open output early if requested (stream writes)
    outf = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        outf = open(args.out, "w", encoding="utf-8")

    Executor = ProcessPoolExecutor if args.mp else ThreadPoolExecutor

    def process_future(fut):
        nonlocal bad_count, total_checked
        bad_path, err = fut.result()
        total_checked += 1
        if bad_path is not None:
            bad_count += 1
            err_types[err.split(":")[0]] += 1
            line = f"{bad_path}\t{err}\n"
            if outf:
                outf.write(line)
            else:
                # If not writing to file, print the bad line (buffered)
                print(line, end="")
        if total_checked % progress_every == 0:
            elapsed = time.time() - t0
            rate = total_checked / max(elapsed, 1e-6)
            print(f"  …checked {total_checked:,}  |  bad: {bad_count:,}  |  {rate:,.0f} imgs/s")

    try:
        with Executor(max_workers=workers) as ex:
            paths = iter_files(root, exts)
            # Seed a bounded set of futures
            pending = set()
            for _ in range(workers * prefetch):
                try:
                    p = next(paths)
                except StopIteration:
                    break
                pending.add(ex.submit(check_image, p))

            # Main loop: for each new path, wait for one to finish, then submit another
            for p in paths:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    process_future(fut)
                pending.add(ex.submit(check_image, p))

            # Drain remaining
            for fut in wait(pending).done:
                process_future(fut)
    finally:
        if outf:
            outf.flush()
            outf.close()

    elapsed = time.time() - t0
    print("\nSummary:")
    print(f"  Checked     : {total_checked:,}")
    print(f"  Bad files   : {bad_count:,}")
    if err_types:
        print("  Error types :")
        for k, v in err_types.most_common():
            print(f"    {k}: {v}")
    print(f"\nElapsed: {elapsed:.1f}s  |  Throughput: {total_checked / max(elapsed, 1e-6):,.0f} imgs/s")

if __name__ == "__main__":
    main()