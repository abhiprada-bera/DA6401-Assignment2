import json

notebook_path = "c:/Users/AbhipradaBera/Desktop/DA-Assignment2/assignment2.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if "outputs" in cell:
        for output in cell["outputs"]:
            if "image/png" in output.get("data", {}):
                img_data = output["data"]["image/png"]
                # If it's a single long string, split it into chunks of 76 characters
                if isinstance(img_data, str):
                    chunk_size = 76
                    output["data"]["image/png"] = [img_data[i:i+chunk_size] + "\n" for i in range(0, len(img_data), chunk_size)]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Reformatted base64 images into chunks.")
