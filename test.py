import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

from model_1.model_1 import ScrewClassifier as ScrewClassifier1
from model_2.model_2 import ScrewClassifier as ScrewClassifier2
from model_3.model_3 import ScrewClassifier as ScrewClassifier3

BASE_DIR = Path(__file__).resolve().parent

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# unnormalized for model 1
unnormalized_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

normalized_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE
])

def evaluate_model(model_name, model, test_loader, device, classes):
    model.eval()
    correct_by_class = {cls_name: 0 for cls_name in classes}
    total_by_class = {cls_name: 0 for cls_name in classes}
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            total_samples += labels.size(0)
            total_correct += (preds == labels).sum().item()
            
            for label, pred in zip(labels, preds):
                cls_name = classes[label.item()]
                if label == pred:
                    correct_by_class[cls_name] += 1
                total_by_class[cls_name] += 1
                
    overall_acc = 100 * total_correct / total_samples
    print(f"\n==========================================")
    print(f"       RESULTS FOR {model_name.upper()}")
    print(f"==========================================")
    print(f"Overall Test Accuracy: {overall_acc:.2f}%\n")
    print("Per-Class Accuracy Breakdown:")
    for cls_name in classes:
        cnt = total_by_class[cls_name]
        acc = (100 * correct_by_class[cls_name] / cnt) if cnt > 0 else 0
        print(f"  - {cls_name:15s}: {acc:6.2f}% ({correct_by_class[cls_name]}/{cnt})")
    
    return overall_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Evaluation on Device: {device}\n")
    
    # images for model 1
    unnorm_test_data = datasets.ImageFolder(BASE_DIR / 'dataset' / 'test', transform=unnormalized_transform)
    unnorm_test_loader = DataLoader(unnorm_test_data, batch_size=32, shuffle=False)
    classes = unnorm_test_data.classes
    
    # images for models 2 & 3
    norm_test_data = datasets.ImageFolder(BASE_DIR / 'dataset' / 'test', transform=normalized_transform)
    norm_test_loader = DataLoader(norm_test_data, batch_size=32, shuffle=False)
    
    models = {
        "Model 1": (ScrewClassifier1(num_classes=5), BASE_DIR / "model_1" / "screw_classifier_1.pth", unnorm_test_loader),
        "Model 2": (ScrewClassifier2(num_classes=5), BASE_DIR / "model_2" / "screw_classifier_2.pth", norm_test_loader),
        "Model 3": (ScrewClassifier3(num_classes=5), BASE_DIR / "model_3" / "screw_classifier_3.pth", norm_test_loader),
    }
    
    results = {}
    for name, (model, path, test_loader) in models.items():
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            acc = evaluate_model(name, model, test_loader, device, classes)
            results[name] = acc
        else:
            print(f"\n[WARNING] Checkpoint not found for {name} at {path}")
            
    print("\n==========================================")
    print("       FINAL MODEL COMPARISON SUMMARY     ")
    print("==========================================")
    for name, acc in results.items():
        print(f"  {name:30s}: {acc:.2f}%")

if __name__ == "__main__":
    main()
