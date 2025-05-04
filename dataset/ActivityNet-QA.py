import os
import subprocess
from pathlib import Path
import zipfile

# Set working directories
work_dir = Path("/home/zye52/scr4_hlee283/zye52/ActivityNet-QA")
zip_dir = work_dir / "zips"
out_dir = work_dir 

zip_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

# Download videos_chunked_01.zip to videos_chunked_29.zip
for i in range(1, 29):
    index = f"{i:02d}"
    file_name = f"videos_chunked_{index}.zip"
    file_path = zip_dir / file_name
    url = f"https://huggingface.co/datasets/lmms-lab/Video-MME/resolve/main/{file_name}"

    if file_path.exists():
        print(f"✅ Already exists: {file_name}")
    else:
        print(f"⬇️ Downloading: {file_name}")
        subprocess.run(["wget", "-c", url, "-O", str(file_path)], check=True)

# Extract all zip files
for zip_file in zip_dir.glob("videos_chunked_*.zip"):
    print(f"📦 Extracting: {zip_file.name}")
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(out_dir)

print(f"✅ All videos extracted to: {out_dir}")
