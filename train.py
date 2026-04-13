import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MultiTaskPerceptionModel
from losses import CustomIoULoss
from data.pets_dataset import PetDataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
BATCH_SIZE = 16
LR = 3e-4

def train_one_epoch(model, loader, optimizer, criteria, device):
    model.train()
    total_loss = 0
    for imgs, labels, bboxes, masks in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits, pred_bboxes, pred_masks = model(imgs)
        
        # Multi-task loss calculation
        loss_cls = criteria['cls'](logits, labels)
        loss_loc = criteria['loc'](pred_bboxes, bboxes)
        loss_seg = criteria['seg'](pred_masks, masks)
        
        # Weighted sum: 1.0, 1.0, 1.0 (can be tuned)
        loss = loss_cls + loss_loc + loss_seg
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Initialize Model
    model = MultiTaskPerceptionModel().to(DEVICE)
    
    # Dataset & Dataloader
    train_ds = PetDataset(split='train')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criteria = {
        'cls': nn.CrossEntropyLoss(),
        'loc': CustomIoULoss(),
        'seg': nn.CrossEntropyLoss()
    }
    
    print(f"Starting training on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criteria, DEVICE)
        print(f"Epoch {epoch}/{EPOCHS} -> Loss: {loss:.4f}")
        
    # Save final model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/multitask_final.pth')
    print("Training complete. Model saved to checkpoints/multitask_final.pth")

if __name__ == "__main__":
    main()
