# Screw Defect Detection
This project uses a PyTorch computer vision pipeline to classify various defects in images of screws. I started this project to introduce myself to PyTorch, and to get started with machine learning.

## Overview
Identifying defects in products is incredibly important in manufacturing. I implemented a Convolutional Neural Network (CNN) in PyTorch to classify images of screws based on their defects. The project is split into 4 models, which use different techniques to reach higher classification accuracy.

## Dataset
This project was created using the [Screw Defect Instance Segmentation Dataset](https://www.kaggle.com/datasets/mahboobxalam/screw-defect-instance-segmentation-dataset?resource=download) from Kaggle. This Dataset contains 737 images at 244x244 pixels sorted into 5 classes:
- head_defect: Cracks, deformation, or wear on the screw head.
- neck_defect: Fractures or thinning at the head-shank interface.
- screw: Defect-free, baseline healthy screws.
- thread_defect: Stripped, malformed, or worn helical threads.
- tip_defect: Broken or blunt screw entry points.

The dataset is pre-split into training (70%), validation (20%), and testing (10%) partitions.

## Models

### Model 1
Model 1 was my introduction to PyTorch. The model uses two convolutional layers to process the images down to a spatial resolution of 56x56 with 32 channels.

### Model 2
Model 2 added normalization to images and would transform them in random ways (rotation, flipping). Normalization will make calculating the gradients easier, while transforming the training images randomly would make it so the model could not memorize the training set, reducing overfit.

### Model 3
Model 3 included the changes from model 2, but also increased the amount of convolutional layers to four, and added a dropout rate of 0.4. Adding more layers allows the model to extract deeper features and increases accuracy. Dropout turns off a percentage of the neurons(40%) during training. This decreases the dependency that neurons have on each other. By not relying too much on specific neurons, the model will have less overfit.

### Model 4
Model 4 uses transfer learning with the ResNet18 CNN, which is trained on the ImageNet database. By leveraging the neural connections made on a model trained on over 1.2 million images, model 4 can adapt the high level feature maps to the screw image dataset. This makes it much faster to train a higher accuracy model as opposed to training solely from the relatively limited dataset the other models were trained on.

## Results

### Model Performance Comparison

| Model | Peak Validation Accuracy | Final Test Accuracy | Net Test Gain vs Baseline |
| :--- | :--- | :---: | :---: |
| **Model 1** | 46.94% | 42.67% | Base Reference |
| **Model 2** | 43.54% | 50.67% | +8.00% |
| **Model 3** | 61.22% | 64.00% | +21.33% |
| **Model 4** | 78.23% | 66.67% | +24.00% |

### Model 3 and Model 4 Breakdown

#### Model 3
- head_defect    :  58.33% (7/12)
- neck_defect    :  57.14% (8/14)
- screw          :  75.00% (15/20)
- thread_defect  :  58.82% (10/17)
- tip_defect     :  66.67% (8/12)

#### Model 4
- head_defect    :  75.00% (9/12)
- neck_defect    :  64.29% (9/14)
- screw          :  85.00% (17/20)
- thread_defect  :  64.71% (11/17)
- tip_defect     :  33.33% (4/12)

While Model 4 reached higher accuracy overall, it appears that it suffers on detecting tip defects. While model 4 reached a much higher accuracy during validation than model 3, on the actual testing it only did about 3% better, which is a suprising result. 

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Anukula-Kashyap/Screw-Defect-Detection-.git
cd Screw-Defect-Detection-
```

### 2. Dataset Setup
1. Download the [Screw Defect Instance Segmentation Dataset](https://www.kaggle.com/datasets/mahboobxalam/screw-defect-instance-segmentation-dataset?resource=download) from Kaggle.
2. Extract the dataset into an `archive/` folder in the project root:
   ```text
   Screw-Defect-Detection-/
   ├── archive/
   │   └── screw_defect_dataset/
   │       ├── images/
   │       └── labels/
   ```

### 3. Sort & Prepare Dataset
Run `data_sort.py` to parse YOLO label annotations and automatically sort images into PyTorch `ImageFolder` format (`dataset/{train,val,test}/{class_name}`):
```bash
python data_sort.py
```

### 4. Train a Model
Train any of the 4 models (e.g. Model 4):
```bash
python model_4/train_4.py
```

### 5. Evaluate All Models
Run `test.py` to generate the side-by-side comparative test evaluation across all models:
```bash
python test.py
```
