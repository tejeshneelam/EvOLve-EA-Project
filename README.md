# 🧠 Alzheimer's Disease Detection using MRI & PET Fusion with Evolutionary Optimization

## 📌 Overview

This project focuses on early detection of Alzheimer's disease using a
multimodal deep learning approach. We combine MRI (structural) and PET
(functional) brain imaging data and apply feature-level fusion optimized
using Differential Evolution (DE).

------------------------------------------------------------------------

## 🎯 Objectives

-   Extract features from MRI and PET images
-   Perform multimodal fusion
-   Optimize fusion using DE
-   Improve classification accuracy

------------------------------------------------------------------------

## 🏗️ Architecture

MRI → CNN → Features\
PET → CNN → Features\
Fusion → DE Optimization → Classifier

------------------------------------------------------------------------

## 📂 Dataset

-   Public Alzheimer datasets (Kaggle)
-   Classes:
    -   NonDemented
    -   VeryMildDemented
    -   MildDemented
    -   ModerateDemented

------------------------------------------------------------------------

## ⚙️ Methodology

-   ResNet50 (pretrained)
-   Feature extraction
-   Fusion (average + DE optimized)
-   Logistic Regression classifier

------------------------------------------------------------------------

## 📊 Results

-   MRI: \~68%
-   PET: \~76%
-   Fusion: \~97%
-   DE Fusion: \~98.6%

------------------------------------------------------------------------

## 📈 Metrics

-   Accuracy
-   F1-score
-   AUC
-   Confusion Matrix
-   Cross-validation

------------------------------------------------------------------------

## 🚀 Run

pip install -r requirements.txt\
python train.py

------------------------------------------------------------------------

## 👨‍💻 Author

Tejesh Neelam
