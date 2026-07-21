import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=5, freeze_features=True):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
            
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_model(num_classes=5, freeze_features=True)
    dummy_batch = torch.randn(32, 3, 224, 224)
    predictions = model(dummy_batch)
    print(f"ResNet-18 Transfer Learning Output shape: {predictions.shape}")
