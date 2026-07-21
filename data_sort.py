import os
import shutil
from pathlib import Path

BASE_DIR = Path("./archive/screw_defect_dataset")
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
OUTPUT_DIR = Path("./dataset")

CLASS_NAMES = {
    0: 'head_defect',
    1: 'neck_defect',
    2: 'screw',
    3: 'thread_defect',
    4: 'tip_defect'
}

DEFECT_CLASSES = {0, 1, 3, 4}

def get_primary_class(label_path: Path) -> str:
    if not label_path.exists():
        return 'screw'  # Default if label file is missing
        
    class_ids = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_ids.append(int(parts[0]))
                
    if not class_ids:
        return 'screw'
        
    
    defects = [cid for cid in class_ids if cid in DEFECT_CLASSES]
    if defects:
        # pick most frequent defect
        primary_id = max(set(defects), key=defects.count)
    else:
        primary_id = 2  
        
    return CLASS_NAMES.get(primary_id, 'screw')

def process_data():
    splits = ['train', 'val', 'test']
    
    for split in splits:
        for cls_name in CLASS_NAMES.values():
            (OUTPUT_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)
            
    stats = {split: {cls: 0 for cls in CLASS_NAMES.values()} for split in splits}
    
    
    for split in splits:
        split_img_dir = IMAGES_DIR / split
        split_lbl_dir = LABELS_DIR / split
        
        if not split_img_dir.exists():
            print(f"Directory not found: {split_img_dir}")
            continue
            
        for img_file in split_img_dir.iterdir():
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            label_file = split_lbl_dir / f"{img_file.stem}.txt"
            class_name = get_primary_class(label_file)
            
            dest_path = OUTPUT_DIR / split / class_name / img_file.name
            shutil.copy2(img_file, dest_path)
            stats[split][class_name] += 1
            
    print(f"\nSorted Dataset into {OUTPUT_DIR.resolve()}:")
    for split in splits:
        print(f"\n--- {split.upper()} ---")
        for cls_name, count in stats[split].items():
            print(f"  {cls_name}: {count} images")

if __name__ == "__main__":
    process_data()