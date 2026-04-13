import json

notebook_path = "assignment2.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            if line.startswith("===") or line.startswith("Cell "):
                line = "# " + line
            new_source.append(line)
        cell["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Fixed syntax issues in notebook cells.")
