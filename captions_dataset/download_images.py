#!/usr/bin/env python3
import csv, os, sys
import requests
from io import BytesIO
from urllib.parse import urlparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def download_and_resize(idx, url, output_dir, session, timeout=10, start_from_id=0):
    if idx < start_from_id:
        return
    # Compute output path early
    out_name = f"{idx:07d}.jpg"
    out_path = os.path.join(output_dir, out_name)

    # Skip if already downloaded
    if os.path.exists(out_path):
        print(f"[{idx:07d}] already exists, skipping")
        return

    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        print(f"[{idx:07d}] ERROR downloading or opening: {e}")
        return

    w, h = img.size
    side = min(w, h)
    crop = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    resized = crop.resize((448, 448), Image.LANCZOS)

    try:
        resized.save(out_path, 'JPEG', quality=90)
        print(f"[{idx:07d}] saved")
    except Exception as e:
        print(f"[{idx:07d}] ERROR saving: {e}")

def main(tsv_path, output_dir, start_from_id, max_workers=1024):
    os.makedirs(output_dir, exist_ok=True)
    # Load URLs into memory
    with open(tsv_path, newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f, delimiter='\t'))
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=max_workers) as exec:
        futures = []
        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue
            url = row[1]
            futures.append(exec.submit(download_and_resize, idx, url, output_dir, session, start_from_id=start_from_id))
        # Wait for all to finish
        for _ in futures:
            _.result()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} TSV_FILE OUTPUT_DIR")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))