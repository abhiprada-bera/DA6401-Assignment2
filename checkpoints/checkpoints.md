# Model Checkpoints

This directory is intended to store the trained weights for the Multi-Task Perception Model.

### Available Checkpoints

The base `MultiTaskPerceptionModel` class is configured to automatically download pre-trained weights from Google Drive during initialization. These weights are saved to the following locations by default:

- `saved_models/classifier.pth`: Classification weights
- `saved_models/localizer.pth`: Bounding box regression weights
- `saved_models/unet.pth`: Segmentation weights

### Local Training Checkpoints

When running `train.py`, a unified checkpoint will be saved to:
- `checkpoints/multitask_final.pth`

> [!TIP]
> To load a local checkpoint, initialize the model and use `torch.load()`:
> ```python
> model = MultiTaskPerceptionModel()
> model.load_state_dict(torch.load('checkpoints/multitask_final.pth'))
> ```
