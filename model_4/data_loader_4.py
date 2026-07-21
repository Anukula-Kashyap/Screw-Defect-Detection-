import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Standard ImageNet normalization
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    NORMALIZE
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE
])

train_data = datasets.ImageFolder(BASE_DIR / 'train', transform=train_transform)
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
