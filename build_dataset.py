import os, sys, csv, random, shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

BASE_DIR  = Path(os.path.abspath(__file__)).parent
DATA_DIR  = BASE_DIR / "data"
IMG_DIR   = BASE_DIR / "images"
ANNOT_DIR = BASE_DIR / "annotations"
MASK_DIR  = ANNOT_DIR / "trimaps"
XML_DIR   = ANNOT_DIR / "xmls"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42

random.seed(SEED)

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

print("Parsing annotations/list.txt...")
all_entries = parse_list(ANNOT_DIR / "list.txt")

for e in all_entries:
    e["xmin"], e["ymin"], e["xmax"], e["ymax"] = parse_bbox(XML_DIR / f"{e['filename']}.xml")

valid = [e for e in all_entries if (IMG_DIR / f"{e['filename']}.jpg").exists()]
print(f"Valid image samples found: {len(valid)} / {len(all_entries)}")

print("Creating stratified split...")
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
        
        if si.exists(): 
            shutil.copy2(si, img_out  / f"{f}.jpg")
            ni += 1
        mask_path = ""
        if sm.exists(): 
            shutil.copy2(sm, mask_out / f"{f}.png")
            nm += 1
            
        rows.append({c: e[c] for c in CSV_COLS})
        
    with open(DATA_DIR / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"  [{name:5s}] images {ni}/{len(entries)} \t masks {nm}/{len(entries)} \t -> {name}.csv")

print("Copying files to data/ directory structure...")
copy_split(train_set, "train")
copy_split(val_set,   "val")
copy_split(test_set,  "test")

print("SUCCESS: Dataset is structured and ready!")
