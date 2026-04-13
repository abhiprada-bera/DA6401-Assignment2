import json

nb = json.load(open("assignment2.ipynb", encoding="utf-8"))
for i, c in enumerate(nb["cells"][:40]):
    src = "".join(c.get("source", []))
    if "QuickDS" in src or "split_/" in src:
        # isolate just the beginning
        preview = src[:300].encode("ascii", "replace").decode("ascii")
        print("Cell %d (%s): %s" % (i, c["cell_type"], preview[:200]))
