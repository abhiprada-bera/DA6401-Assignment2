import torch
from PIL import Image
from models import MultiTaskPerceptionModel
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

def load_model(checkpoint_path):
    model = MultiTaskPerceptionModel()
    # If using local weights after training
    # model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(model, image_path):
    img = Image.open(image_path).convert("RGB")
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    input_tensor = tfm(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits, bboxes, masks = model(input_tensor)
        
    return logits, bboxes, masks

def main():
    # Example usage
    # model = load_model('checkpoints/multitask_final.pth')
    # Use weights downloaded during init
    model = MultiTaskPerceptionModel().to(DEVICE).eval()
    
    # print("Model loaded and ready for inference.")
    # result = predict(model, 'path/to/your/pet.jpg')

if __name__ == "__main__":
    main()
