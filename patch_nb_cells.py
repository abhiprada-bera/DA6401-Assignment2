import json, nbformat as nbf

nb = json.load(open("assignment2.ipynb", encoding="utf-8"))
print("Total cells before:", len(nb["cells"]))

# Fix cells 24, 27, 37 which have the old broken QuickDS with split_/
# Replace the broken path pattern in their source
BAD  = "f\"{DATA_DIR}/{self.split_}/{s['filename']}.jpg\""
GOOD = "f\"{DATA_DIR}/{self.split_name}/images/{s['filename']}.jpg\""

BAD_PROP = "@property\n    def split_(self): return \"train\"\n\nclass QuickDSSplit(QuickDS):\n    def __init__(self, split, n=128):\n        self.split_name = split\n        super().__init__(split, n)\n    @property\n    def split_(self): return self.split_name"
GOOD_ALIAS = "\nQuickDSSplit = QuickDS  # alias"

fixed = 0
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if BAD in src:
        print(f"Fixing cell {i}: replacing bad path")
        new_src = src.replace(BAD, GOOD)
        new_src = new_src.replace(BAD_PROP, GOOD_ALIAS)
        if isinstance(c["source"], list):
            c["source"] = [new_src]
        else:
            c["source"] = new_src
        fixed += 1

print(f"Fixed {fixed} cells")

with open("assignment2.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Saved.")
