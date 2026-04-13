import json
import re

def process_source(text):
    text = re.sub(r'import albumentations as A\nfrom albumentations\.pytorch import ToTensorV2', 
                  'import torchvision.transforms as T\nfrom torchvision.transforms import functional as F', text)
    
    # 2.1 Detection Dataset
    # Replace the initialization
    text = re.sub(r'if self\.transform:\n\s*img = self\.transform\(image=img\)\["image"\]',
                  'if self.transform:\n            img = self.transform(img)', text)
    # img = np.array(Image.open(s["path"]).convert("RGB")) -> img = Image.open(s["path"]).convert("RGB")
    text = re.sub(r'img = np\.array\(Image\.open\(s\["path"\]\)\.convert\("RGB"\)\)',
                  'img = Image.open(s["path"]).convert("RGB")', text)
    text = re.sub(r'oh, ow = img\.shape\[:2\]', 'ow, oh = img.size', text)
    
    # Det transforms
    det_trans_old = """det_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])
det_val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])"""
    det_trans_new = """det_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])
det_val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])"""
    text = text.replace(det_trans_old, det_trans_new)
    
    # 2.5 Segmentation Dataset
    text = re.sub(r'img\s*=\s*np\.array\(Image\.open\(s\["img"\]\)\.convert\("RGB"\)\)',
                  'img = Image.open(s["img"]).convert("RGB")', text)
    text = re.sub(r'mask = np\.array\(Image\.open\(s\["mask"\]\)\)',
                  'mask = Image.open(s["mask"])', text)
    # Replace the clipping and transform block
    seg_get_old = """        mask = np.clip(mask - 1, 0, 2).astype(np.uint8)  # 1,2,3 → 0,1,2
        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return img, mask.long()"""
    seg_get_new = """        import numpy as np
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img = T.ToTensor()(img)
        img = T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))(img)
        mask = np.array(mask)
        mask = np.clip(mask - 1, 0, 2).astype(np.int64)
        return img, torch.tensor(mask, dtype=torch.long)"""
    text = text.replace(seg_get_old, seg_get_new)
    
    seg_trans_old = """seg_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])"""
    text = text.replace(seg_trans_old, "seg_transform = None")
    
    # Det epochs fix if needed
    text = text.replace("DET_EPOCHS = 15", "DET_EPOCHS = 1")
    text = text.replace("SEG_EPOCHS = 15", "SEG_EPOCHS = 1")
    
    return text

with open("assignment2_part4.py", "r", encoding="utf-8") as f:
    text = f.read()

text = process_source(text)

with open("assignment2_part4.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated assignment2_part4.py")

nb_path = "assignment2.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        new_source = process_source(source)
        # convert string back to list of lines
        if source != new_source:
             lines = [line + '\n' for line in new_source.split('\n')]
             # remove trailing newline from last line
             if lines and lines[-1] == '\n':
                 lines.pop()
             elif lines:
                 lines[-1] = lines[-1][:-1]
             cell["source"] = lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Updated notebook cells")
