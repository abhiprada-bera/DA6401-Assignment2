"""
Oxford-IIIT Pet Dataset - Download & Split (with retry + integrity check)
=========================================================================
- Validates tar.gz integrity before skipping re-download
- Retries up to MAX_RETRIES times with back-off on connection errors

Output:
  data/
    train/  images/  masks/
    val/    images/  masks/
    test/   images/  masks/
    train.csv  val.csv  test.csv
"""

import sys, shutil, csv, random, tarfile, time
import xml.etree.ElementTree as ET
import urllib.request, urllib.error
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"

IMAGES_URL = "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz"
ANNOT_URL  = "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz"
IMAGES_TAR = RAW_DIR / "images.tar.gz"
ANNOT_TAR  = RAW_DIR / "annotations.tar.gz"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42
MAX_RETRIES = 15
CHUNK       = 1024 * 512   # 512 KB chunks

random.seed(SEED)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helpers ----------------------------------------------------------------
def is_valid_tar(path: Path) -> bool:
    """Quick integrity check - reads the tar TOC without extracting."""
    try:
        with tarfile.open(path, "r:gz") as t:
            t.getmembers()
        return True
    except Exception:
        return False

def download_with_retry(url: str, dest: Path):
    if dest.exists() and is_valid_tar(dest):
        print(f"  [OK] {dest.name} already complete and valid, skipping.")
        return
    if dest.exists():
        print(f"  [!] {dest.name} incomplete/corrupt, removing and re-downloading.")
        dest.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [>>] Attempt {attempt}: downloading {dest.name} ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                written = 0
                with open(dest, "wb") as fh:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        if total:
                            pct = min(100, written * 100 // total)
                            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
                            print(f"\r    [{bar}] {pct:3d}%  "
                                  f"{written/1e6:7.1f}/{total/1e6:.1f} MB",
                                  end="", flush=True)
            print()
            if is_valid_tar(dest):
                print(f"  [OK] {dest.name} downloaded and verified.")
                return
            else:
                print(f"  [!] Downloaded file failed integrity check, retrying ...")
                dest.unlink()
        except Exception as e:
            print(f"\n  [WARN] Attempt {attempt} failed: {type(e).__name__}: {e}")
            if dest.exists():
                dest.unlink()
            wait = min(30, 5 * attempt)
            print(f"         Waiting {wait}s before retry ...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts.")

def extract_if_needed(tar_path: Path, dest_dir: Path, sentinel: Path):
    if sentinel.exists():
        print(f"  [OK] Already extracted: {sentinel}")
        return
    print(f"  [>>] Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(dest_dir)
    print(f"       Done -> {dest_dir}")

# ---- 1. Download ------------------------------------------------------------
print("\n=== STEP 1: Download Oxford-IIIT Pet Dataset ===")
download_with_retry(IMAGES_URL, IMAGES_TAR)
download_with_retry(ANNOT_URL,  ANNOT_TAR)

# ---- 2. Extract -------------------------------------------------------------
print("\n=== STEP 2: Extract archives ===")
IMG_DIR   = RAW_DIR / "images"
ANNOT_DIR = RAW_DIR / "annotations"
extract_if_needed(IMAGES_TAR, RAW_DIR, IMG_DIR)
extract_if_needed(ANNOT_TAR,  RAW_DIR, ANNOT_DIR)

MASK_DIR = ANNOT_DIR / "trimaps"
XML_DIR  = ANNOT_DIR / "xmls"

# ---- 3. Parse list.txt ------------------------------------------------------
print("\n=== STEP 3: Parse annotations ===")

def parse_list(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            entries.append({
                "filename"   : parts[0],
                "breed_label": int(parts[1]) - 1,
                "species"    : int(parts[2]) - 1,
            })
    return entries

all_entries = parse_list(ANNOT_DIR / "list.txt")
print(f"  Total entries : {len(all_entries)}")

def parse_bbox(xml_path):
    if not xml_path.exists():
        return -1, -1, -1, -1
    try:
        root = ET.parse(xml_path).getroot()
        obj  = root.find("object")
        if obj is None:
            return -1, -1, -1, -1
        bb = obj.find("bndbox")
        return (int(float(bb.find("xmin").text)),
                int(float(bb.find("ymin").text)),
                int(float(bb.find("xmax").text)),
                int(float(bb.find("ymax").text)))
    except Exception:
        return -1, -1, -1, -1

for e in all_entries:
    e["xmin"], e["ymin"], e["xmax"], e["ymax"] = parse_bbox(XML_DIR / f"{e['filename']}.xml")

valid = [e for e in all_entries if (IMG_DIR / f"{e['filename']}.jpg").exists()]
print(f"  Valid samples : {len(valid)}")

# ---- 4. Stratified split ----------------------------------------------------
print("\n=== STEP 4: Stratified split (70/15/15) ===")
buckets = defaultdict(list)
for e in valid:
    buckets[e["breed_label"]].append(e)

train_set, val_set, test_set = [], [], []
for _, items in sorted(buckets.items()):
    random.shuffle(items)
    n = len(items)
    n_tr = max(1, int(n * TRAIN_RATIO))
    n_va = max(1, int(n * VAL_RATIO))
    train_set.extend(items[:n_tr])
    val_set.extend(  items[n_tr : n_tr + n_va])
    test_set.extend( items[n_tr + n_va :])

random.shuffle(train_set)
random.shuffle(val_set)
random.shuffle(test_set)
print(f"  train {len(train_set)}  |  val {len(val_set)}  |  test {len(test_set)}")

# ---- 5. Copy + CSV ----------------------------------------------------------
print("\n=== STEP 5: Copy files and write CSVs ===")
CSV_COLS = ["filename","breed_label","species","xmin","ymin","xmax","ymax"]

def copy_split(entries, name):
    img_out  = DATA_DIR / name / "images"
    mask_out = DATA_DIR / name / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    rows = []
    ni = nm = 0
    for e in entries:
        f = e["filename"]
        si = IMG_DIR  / f"{f}.jpg"
        sm = MASK_DIR / f"{f}.png"
        if si.exists(): shutil.copy2(si, img_out  / f"{f}.jpg"); ni += 1
        if sm.exists(): shutil.copy2(sm, mask_out / f"{f}.png"); nm += 1
        rows.append({c: e[c] for c in CSV_COLS})
    with open(DATA_DIR / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader(); w.writerows(rows)
    print(f"  [{name:5s}]  images {ni}/{len(entries)}  masks {nm}/{len(entries)}  -> {name}.csv")

copy_split(train_set, "train")
copy_split(val_set,   "val")
copy_split(test_set,  "test")

print(f"""
[DONE]
  data/train/  {len(train_set)} samples
  data/val/    {len(val_set)} samples
  data/test/   {len(test_set)} samples
  CSVs: train.csv  val.csv  test.csv
""")
