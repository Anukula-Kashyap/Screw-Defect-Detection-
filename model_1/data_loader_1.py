import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()           
])

train_data = datasets.ImageFolder(BASE_DIR / 'train', transform=transform)
val_data = datasets.ImageFolder(BASE_DIR / 'val', transform=transform)

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    print(f"Classes found: {train_data.classes}")
    print(f"Total training batches: {len(train_loader)}")
    print(f"Total validation batches: {len(val_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
