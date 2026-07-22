# Screw Defect Detection
This project uses a PyTorch computer vision pipeline to classify various defects in images of screws. I started this project to introduce myself to PyTorch, and to get started with machine learning.

## Overview
Identifying defects in products is incredibly important in maufacturing. I implemented a Convolutional Neural Network(CNN) in PyTorch to classify imagess of screws based on their defects. The project is split into 4 models, which use different techniques to reach higher classification accuracy.

## Dataset
This project was created using the [Screw Defect Instance Segmentation Dataset](https://www.kaggle.com/datasets/mahboobxalam/screw-defect-instance-segmentation-dataset?resource=download) from Kaggle. This Dataset contains 737 images at 244x244 pixels sorted into 5 classes:
- head_defect: Cracks, deformation, or wear on the screw head.
- neck_defect: Fractures or thinning at the head-shank interface.
- screw: Defect-free, baseline healthy screws.
- thread_defect: Stripped, malformed, or worn helical threads.
- tip_defect: Broken or blunt screw entry points.

The dataset is pre-split into training(70%), validation(20%), and testing(10%) partitions.

## Models

### Model 1
Model 1 was my introduction to PyTorch. The model uses two convolutional layers to process the images down to a spatial resolution of 56x56 with 32 channels.

### Model 2
Model 2 added normalization to images and would transform them in random ways (rotation, flipping). Normalization will make calculating the gradients easier, while transforming the training images randomly would make it so the model could not memorize the training set, reducing overfit.

### Model 3
Model 3 included the changes from model 2, but also increased the amount of convolutional layers to four, and added a dropout rate of 0.4. adding more layers allows the model to extract deeper features and increases accuracy. Dropout turns off a percentage of the neurons(40%) during training. This decreases the dependency that neurons have on each other. By not relying too much on specific neurons, the model will have less overfit.

### Model 4
Model 4 uses transfer learning with the ResNet18 CNN, which is trained on the ImageNet database. By leveraging the neural connections made on a model trained on over 1.2 million images, model 4 can adapt the high level feature maps to the screw image dataset. This makes it much faster to train a higher accuracy model as opposed to training solely from the relatively limited dataset the other models were trained on.

##Results



