import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class PetDataset(Dataset):
    """
    Unified dataset class for Oxford-IIIT Pet tasks.
    Supports classification, bounding box regression, and segmentation.
    """
    def __init__(self, split, data_dir='data', img_size=224, transform=None, n=None):
        self.split    = split
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.rows     = []
        
        csv_path = os.path.join(data_dir, f"{split}.csv")
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                img_path = os.path.join(data_dir, split, "images", f"{r['filename']}.jpg")
                mask_path = os.path.join(data_dir, split, "masks", f"{r['filename']}.png")
                
                if os.path.exists(img_path):
                    self.rows.append({
                        "filename": r["filename"],
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "breed_label": int(r["breed_label"]),
                        "xmin": int(r["xmin"]), "ymin": int(r["ymin"]),
                        "xmax": int(r["xmax"]), "ymax": int(r["ymax"]),
                    })
        if n: self.rows = self.rows[:n]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        img = Image.open(item["img_path"]).convert("RGB")
        ow, oh = img.size
        
        # Default transforms if none provided
        if self.transform:
            img_t = self.transform(img)
        else:
            tfm = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            img_t = tfm(img)

        # Classification label
        label = item["breed_label"]

        # Bounding box (normalised)
        if item["xmin"] != -1:
            bbox = torch.tensor([item["xmin"]/ow, item["ymin"]/oh, 
                                 item["xmax"]/ow, item["ymax"]/oh], dtype=torch.float32)
        else:
            bbox = torch.zeros(4, dtype=torch.float32)

        # Segmentation mask
        if os.path.exists(item["mask_path"]):
            mask = Image.open(item["mask_path"])
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_t = torch.tensor((np.array(mask)-1).clip(0,2), dtype=torch.long)
        else:
            mask_t = torch.zeros((self.img_size, self.img_size), dtype=torch.long)

        return img_t, label, bbox, mask_t
