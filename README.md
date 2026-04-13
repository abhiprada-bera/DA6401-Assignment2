# DA6401-Assignment2: Multi-Task Visual Perception Pipeline

This project implements a comprehensive visual perception pipeline using the Oxford-IIIT Pet dataset. It features a unified architecture based on **VGG11-BN** that simultaneously performs three tasks:

1.  **Pet Breed Classification** (37 classes)
2.  **Object Detection** (Bounding box regression)
3.  **Semantic Segmentation** (Trimap-based segmentation)

## Project Structure

```text
.
├── checkpoints/       # Model weights (gitignored)
├── data/             
│   └── pets_dataset.py # Unified PyTorch Dataset class
├── losses/
│   └── iou_loss.py    # Custom IoU loss for localization
├── models/
│   ├── vgg11.py       # VGG11 backbone implementation
│   ├── layers.py      # Custom Dropout and Conv helpers
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py # U-Net style decoder
│   └── multitask.py   # Integrated Multi-Task model
├── train.py           # Unified training script
├── inference.py       # Inference and showcase utility
└── requirements.txt
```

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/abhiprada-bera/DA6401-Assignment2.git
    cd DA6401-Assignment2
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Weight Checkpoints**:
    The `MultiTaskPerceptionModel` is configured to automatically download pre-trained weights from Google Drive on initialization using `gdown`.

## Usage

### Training
To train the unified pipeline:
```bash
python train.py
```

### Inference
To run inference on a sample image:
```bash
python inference.py
```

## Architecture Details
The pipeline uses a shared VGG11-BN backbone. The task-specific heads are:
- **Classification Head**: Global average pooling followed by an MLP with Custom Dropout.
- **Bounding Box Head**: MLP with Sigmoid activation for coordinate regression.
- **Segmentation Head**: U-Net style expansive path with skip connections from the backbone.

## Data
The project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Ensure the images and annotations are placed in the `data/` directory (not included in the repository due to size limitations).
