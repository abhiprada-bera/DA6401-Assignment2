import json

nb_path = "assignment2.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        source = "".join(source_lines)
        if "train_transform = A.Compose" in source:
            old_train = """train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])"""
            new_train = """import torchvision.transforms as T
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])"""
            source = source.replace(old_train, new_train)
            
            old_val = """val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])"""
            new_val = """val_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])"""
            source = source.replace(old_val, new_val)
            
            # fix `if self.transform: img = self.transform(image=img)["image"]` for A
            # but wait, it might be `if self.transform: img = self.transform(img)` in classification dataset
            # Let's verify classification dataset 'PetDataset' implementation in the source string
            if 'if self.transform:\n            img = self.transform(img)' not in source:
                 # it already is just `img = self.transform(img)`
                 pass
            
            lines = [line + '\n' for line in source.split('\n')]
            if lines and lines[-1] == '\n':
                lines.pop()
            elif lines:
                lines[-1] = lines[-1][:-1]
            cell["source"] = lines

        # Also reduce epochs for fast test run if needed?
        # The user said "please run each cells of ipynb file" 
# Just to be safe, I'm finding whether the epochs for VGG11 are changed in the file and ensuring they run quickly or as originally specified. Wait, the user wants the training to run. Changing VGG to 1 epoch is usually expected when asked to run a heavy notebook because training thousands of images on 15 epochs takes hours on cpu, although they may want valid results. I won't touch VGG epochs here because they are not failing.

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Classification transforms patched.")
