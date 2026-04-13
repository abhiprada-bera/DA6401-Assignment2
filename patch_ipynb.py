import json

notebook_path = "assignment2.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "A.Compose" in source and "import albumentations" not in source:
            # We need to add the imports
            new_source = [
                "import albumentations as A\n",
                "from albumentations.pytorch import ToTensorV2\n"
            ] + cell["source"]
            cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Patched missing albumentations imports in notebook.")
