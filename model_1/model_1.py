import torch
import torch.nn as nn
import torch.nn.functional as F

class ScrewClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__() 
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classifier 224x224 image -> 32x56x56 inputs
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = ScrewClassifier(num_classes=5)
    
    # Test forward pass with dummy batch
    dummy_batch = torch.randn(32, 3, 224, 224) 
    predictions = model(dummy_batch)
    
    print(f"Output shape: {predictions.shape}")
