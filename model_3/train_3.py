import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from data_loader_3 import train_loader, val_loader
from model_3 import ScrewClassifier

MODEL_SAVE_PATH = Path(__file__).resolve().parent / "screw_classifier_3.pth"

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Model 3 on device: {device}")
    
    model = ScrewClassifier(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #lr scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    epochs = 50
    patience = 12
    patience_counter = 0
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train() 
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                running_val_loss += val_loss.item()
                
                _, predicted = torch.max(val_outputs, 1) 
                total_samples += val_labels.size(0)
                correct_predictions += (predicted == val_labels).sum().item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        
        scheduler.step(avg_val_loss)
        
        saved_status = ""
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            saved_status = " [model saved]"
        else:
            patience_counter += 1
    
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Val Accuracy: {val_accuracy:.2f}%"
              f"{saved_status}")
              
        if patience_counter >= patience:
            print(f"\n stopped after {epoch+1} epochs. No improvement for {patience} epochs")
            break

    print(f"\nPeak Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
