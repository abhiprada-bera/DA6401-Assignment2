import json
import base64
import os

notebook_path = "c:/Users/AbhipradaBera/Desktop/DA-Assignment2/assignment2.ipynb"
plots_dir = "c:/Users/AbhipradaBera/Desktop/DA-Assignment2/plots"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        matches = []
        if "12_detection_curves.png" in source: matches.append("12_detection_curves.png")
        if "13_bbox_predictions.png" in source: matches.append("13_bbox_predictions.png")
        if "14_segmentation_curves.png" in source: matches.append("14_segmentation_curves.png")
        if "15_segmentation_results.png" in source: matches.append("15_segmentation_results.png")
             
        for match in matches:
            plot_file = os.path.join(plots_dir, match)
            if os.path.exists(plot_file):
                with open(plot_file, "rb") as pf:
                    b64_img = base64.b64encode(pf.read()).decode("utf-8")
                
                output = {
                    "data": {
                        "image/png": b64_img,
                        "text/plain": [f"<Figure size with {match}>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
                
                already_has_match = False
                for existing_out in cell.get("outputs", []):
                    text_plain = existing_out.get("data", {}).get("text/plain", [])
                    if len(text_plain) > 0 and match in text_plain[0]:
                        already_has_match = True
                        break
                
                if not already_has_match:
                    if "outputs" not in cell:
                        cell["outputs"] = []
                    cell["outputs"].append(output)
                    print(f"Added {match} to cell outputs.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Finished updating notebook.")
