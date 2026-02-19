# 🚀 VisionSpec-QC
## Automated PCB Defect Detection using Computer Vision & Deep Learning
### 📌 Overview

VisionSpec-QC is a Computer Vision & Machine Learning project developed to automate PCB (Printed Circuit Board) quality inspection.

The system classifies PCB images into:

✅ PASS

❌ DEFECT

It also integrates Explainable AI (Grad-CAM) to visualize defect regions and improve model interpretability.

### 🎯 Problem Statement

Manual PCB inspection is time-consuming, inconsistent, and error-prone.
This project provides an AI-powered automated solution to improve accuracy, efficiency, and reliability in manufacturing quality control.

### 🧠 Model Architecture

▪️Transfer Learning using MobileNetV2

▪️Convolutional Neural Network (CNN)

▪️Binary Classification (Sigmoid Activation)

▪️Binary Cross-Entropy Loss

### 📊 Dataset

▪️Total Images: 8,500+

▪️Classes: PASS & DEFECT

▪️Addressed class imbalance using class weights

### 🔄 Data Preprocessing

▪️Image resizing (224x224)

▪️Normalization (pixel scaling 0–1)

▪️Data Augmentation:

▪️Rotation

▪️Flipping

▪️Zoom

▪️Width/Height shifting

### 📈 Model Performance

Achieved ~87% training accuracy

Evaluated using:

▪️Accuracy

▪️Loss

▪️Validation Metrics

▪️Confusion Matrix (analysis stage)

### 🔍 Explainable AI

▪️Implemented Grad-CAM (Gradient-weighted Class Activation Mapping) to:

▪️Highlight defect regions

▪️Improve transparency

▪️Increase trust in model predictions

### 🌐 Deployment

▪️Built an interactive Streamlit Web Application

▪️Real-time image upload and prediction

Displays:

PASS / DEFECT result

Confidence score

Grad-CAM heatmap

### 🛠️ Tech Stack

▪️Python

▪️TensorFlow

▪️Keras

▪️OpenCV

▪️NumPy

▪️Matplotlib

▪️Streamlit

▪️CNN

▪️Transfer Learning



VisionSpec-QC/
│
├── data/
├── models/
├── src/
│   ├── train.py
│   ├── gradcam.py
│   ├── test_gradcam.py
│
├── app.py
├── requirements.txt
└── README.md
