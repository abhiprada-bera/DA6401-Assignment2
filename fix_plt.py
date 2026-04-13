import json

nb_path = "assignment2.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source_list = cell.get("source", [])
        new_source_list = []
        for line in source_list:
            if "plt.FancyBboxPatch" in line:
                line = line.replace("plt.FancyBboxPatch", "mpatches.FancyBboxPatch")
                changed = True
            new_source_list.append(line)
        cell["source"] = new_source_list

if changed:
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Fixed plt.FancyBboxPatch")
else:
    print("No changes needed")
