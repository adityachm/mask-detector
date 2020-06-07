# Mask-Detector

## Problem Statement
To detect whether or not a person is wearing a face-mask by using **Convolution Neural Networks** and **OpenCV**

## Dataset 
This dataset contains a `data` folder which contains 2 subdirectories namely
1. `without_mask` - which contains 686 images without mask.
2. `with mask`- which contains  690 images with masks.

## Image Preprocessing
Resizing images to one standard size for Neural network to accept it as input.
Following are preprocessing steps are performed:
* Gray scaling 
* Normalize
* Resize 
* Reshape

## Model Building and Training :
* Custom CNN is built for training since data is not complex to use pretrained models.
* Model is trained on both **with and without masks images**.
* Model is then tested on real world data through webcam feed and giving decent accuracy.

## Model Prediction
The following steps are performed here:
1. Loading the previously saved model(best model) from training phase.
2. Using webcam to generate test images and prediction is made.

