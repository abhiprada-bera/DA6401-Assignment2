import json

nb = json.load(open("assignment2.ipynb", encoding="utf-8"))
cells = nb["cells"]
print(f"Total cells: {len(cells)}")
print("\nCell overview:")
for i, c in enumerate(cells):
    src = "".join(c.get("source", ""))[:70].replace("\n", " ")
    n_out = len(c.get("outputs", []))
    has_img = any("image/png" in o.get("data", {}) for o in c.get("outputs", []))
    has_stream = any(o.get("output_type") == "stream" for o in c.get("outputs", []))
    print(f"  [{i:>2}] {c['cell_type']:<8}  outs={n_out}  img={has_img}  stream={has_stream}  | {src}")
