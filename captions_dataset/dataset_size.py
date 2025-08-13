import os

dir_path = './train'  # ← replace with your folder
# files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
print(f"Found {len(os.listdir(dir_path))} files in {dir_path}")