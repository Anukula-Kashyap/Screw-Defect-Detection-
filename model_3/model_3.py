import torch
import torch.nn as nn
import torch.nn.functional as F

class ScrewClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super().__init__() 
        
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) #extra conv layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 224x224 image -> 128x14x14 inputs
        self.fc1 = nn.Linear(128 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    model = ScrewClassifier(num_classes=5)
    
    # Test forward pass with dummy batch
    dummy_batch = torch.randn(32, 3, 224, 224) 
    predictions = model(dummy_batch)
    
    print(f"Output shape: {predictions.shape}")
