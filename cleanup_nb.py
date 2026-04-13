"""
Clean up the notebook: remove all duplicates,
keep only: Tasks 1.1-1.4 (cells 0-11) + Section 2.5-2.8 + Section 5 (verified executed).
Then generate all Section 2 plots standalone and embed them.
"""
import json, nbformat as nbf

with open("assignment2.ipynb", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

print("Total cells before cleanup:", len(nb.cells))

# Strategy: keep first occurrence of each block, based on unique signatures
# Keep cells 0-11 (Tasks 1.1-1.4) - they have outputs
# Keep cells 12-23 (2.5-2.8 - newly added, no outputs yet)
# Keep first Section 5 block: cells 24-33 (has executed outputs)
# Drop duplicates: everything from 34 onwards that repeats section 2 / section 5

# Identify the first fully-executed Section 5 block
sec5_start = None
sec5_end = None
for i, c in enumerate(nb.cells):
    src = "".join(c.get("source", []))
    if sec5_start is None and ("Section 5" in src and "Automated" in src):
        sec5_start = i
    if sec5_start is not None and sec5_end is None and i > sec5_start:
        # The section ends at the Verification Summary markdown
        if "Verification Summary" in src:
            sec5_end = i
            break

print(f"Section 5 found: cells {sec5_start} to {sec5_end}")

# Build clean cell list: 0..23 (tasks + 2.5-2.8) + sec5_start..sec5_end
clean_cells = nb.cells[0:24] + nb.cells[sec5_start:sec5_end+1]
print(f"Clean cells: {len(clean_cells)}")

nb.cells = clean_cells
with open("assignment2.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook cleaned and saved.")
