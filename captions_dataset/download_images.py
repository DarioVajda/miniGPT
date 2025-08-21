print("Starting imports...")
import csv, os, sys
print("Importing csv, os, sys...")
import requests
print("Importing requests...")
from io import BytesIO
print("Importing BytesIO...")
from urllib.parse import urlparse
print("Importing urlparse...")
from PIL import Image
print("Importing Image from PIL...")
from concurrent.futures import ThreadPoolExecutor
print("Importing ThreadPoolExecutor...")

def download_and_resize(idx, url, output_dir, session, timeout=10, start_from_id=0, final_id=999999999):
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

# def main(tsv_path, output_dir, start_from_id, max_workers=4):
#     os.makedirs(output_dir, exist_ok=True)
#     # Load URLs into memory
#     print(f"Loading URLs from {tsv_path}...")
#     with open(tsv_path, newline='', encoding='utf-8') as f:
#         reader = list(csv.reader(f, delimiter='\t'))
#     print(f"Loaded {len(reader)} entries.")
#     session = requests.Session()
#     with ThreadPoolExecutor(max_workers=max_workers) as exec:
#         futures = []
#         for idx, row in enumerate(reader):
#             if len(row) < 2:
#                 continue
#             url = row[1]
#             futures.append(exec.submit(download_and_resize, idx, url, output_dir, session, start_from_id=start_from_id))
#         # Wait for all to finish
#         for _ in futures:
#             _.result()

def main(tsv_path, output_dir, start_from_id, process_n, max_workers=32):
    os.makedirs(output_dir, exist_ok=True)
    end_id = start_from_id + process_n
    session = requests.Session()  # reuse; cheaper than new per-call

    def iter_jobs():
        with open(tsv_path, newline='', encoding='utf-8') as f:
            for idx, row in enumerate(csv.reader(f, delimiter='\t')):
                if len(row) < 2 or idx < start_from_id:
                    continue
                out_path = os.path.join(output_dir, f"{idx:07d}.jpg")
                if os.path.exists(out_path):  # skip before scheduling
                    print(f"[{idx:07d}] already exists, skipping")
                    continue
                if idx >= end_id + 500:
                    print(f"Stopping at {idx} as it exceeds the limit of {end_id}")
                    print(abc)
                yield idx, row[1]

    def worker(job):
        idx, url = job
        return download_and_resize(
            idx, url, output_dir, session, start_from_id=start_from_id, final_id=(start_from_id + process_n)
        )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # executor.map streams tasks; no huge in-memory lists
        for _ in ex.map(worker, iter_jobs()):
            pass

# if __name__ == '__main__':
print("Starting image download and resize script...")
if len(sys.argv) != 5:
    print(f"Usage: {sys.argv[0]} TSV_FILE OUTPUT_DIR START_FROM_ID NUMBER_OF_EXAMPLES")
    sys.exit(1)
main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))