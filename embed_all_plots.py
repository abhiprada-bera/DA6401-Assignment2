import json
import base64
import os

notebook_path = "assignment2.ipynb"
plots_dir = "plots"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define mapping: if any keyword matches source, we embed the correspoding image
# We must preserve the order if multiple
plot_mappings = {
    "01_class_distribution": "01_class_distribution.png",
    "02_sample_grid": "02_sample_grid_train.png",
    "03_image_stats": "03_image_stats.png",
    "04_bbox_distribution": "04_bbox_distribution.png",
    "05_trimap_samples": "05_trimap_samples.png",
    "06_augmented_batch": "06_augmented_batch.png",
    "07_vgg11_architecture": "07_vgg11_architecture.png",
    "08_training_curves": "08_training_curves.png",
    "09_confusion_matrix": "09_confusion_matrix.png",
    "10_per_class_f1": "10_per_class_f1.png",
    "11_predictions": "11_predictions.png",
    "12_detection_curves": "12_detection_curves.png",
    "13_bbox_predictions": "13_bbox_predictions.png",
    "14_segmentation_curves": "14_segmentation_curves.png",
    "15_segmentation_results": "15_segmentation_results.png",
    "verify_01_vgg_arch"     : "verify_01_vgg_arch.png",
    "verify_02_dropout"      : "verify_02_dropout.png",
    "verify_03_iou_loss"     : "verify_03_iou_loss.png",
    "verify_03b_iou_bar"     : "verify_03b_iou_bar.png",
    "verify_04_pipeline_metrics": "verify_04_pipeline_metrics.png",
}

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        matches = []
        for key, filename in plot_mappings.items():
            if key in source:
                matches.append(filename)
                
        for match in matches:
            plot_file = os.path.join(plots_dir, match)
            if os.path.exists(plot_file):
                with open(plot_file, "rb") as pf:
                    b64_img = base64.b64encode(pf.read()).decode("utf-8")
                
                # Split big base64 string for valid jupyter output format
                chunk_sz = 76
                b64_chunks = [b64_img[i:i+chunk_sz] + "\n" for i in range(0, len(b64_img), chunk_sz)]
                
                output = {
                    "data": {
                        "image/png": b64_chunks,
                        "text/plain": [f"<Figure size with {match}>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
                
                already_has_match = False
                for existing_out in cell.get("outputs", []):
                    # check if image is already there by plain text matching
                    text_plain = existing_out.get("data", {}).get("text/plain", [])
                    if len(text_plain) > 0 and match in text_plain[0]:
                        already_has_match = True
                        break
                    # or if the base64 content is same (rarely we'll need this since text/plain is used clearly)
                
                if not already_has_match:
                    if "outputs" not in cell:
                        cell["outputs"] = []
                    cell["outputs"].append(output)
                    print(f"Added {match} to cell outputs.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Finished updating notebook with all plots.")
