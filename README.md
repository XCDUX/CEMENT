# Ultrasonic Well Segmentation Challenge

## 📌 Project Overview

This repository contains our approach to tackling the **Ultrasonic Well Segmentation Challenge**. The goal of this project is to develop a deep learning model capable of **segmenting ultrasonic well images** to identify key structural interfaces such as **casing, cement, and formation boundaries**. Accurate segmentation is crucial for **well integrity assessment**, ensuring that the cement is properly bonded to the casing and geological formation.

## 🚀 Approach

### 🔹 **Model Architecture**
- We employ a **U-Net architecture** with an **EfficientNet-B3** encoder.
- The encoder is **pretrained on ImageNet** to improve generalization.
- Skip connections allow for high-resolution spatial information retention.

### 🔹 **Data Processing & Augmentation**
- **Reconstructing Well Sections:** Original patches are stacked vertically to restore spatial continuity before being recut with an **80-pixel translation** (data_augm.py).
- **Data Augmentation:** We apply standard transformations such as **flipping, rotation, contrast adjustment, and elastic deformations** to improve robustness (included in the data.py pipeline).
- **Post-Processing:** Physically impossible class predictions are removed (e.g., misclassified well structures on the right side) (post_process.py).

### 🔹 **Self-Supervised Learning**
- We attempted **pseudo-labeling**, where the model predicts on **unlabeled data** and is retrained on the combined **labeled + pseudo-labeled dataset** (SSL.py).
- This approach provided **moderate improvements**, but noise in pseudo-labels remained a challenge.

### 🔹 **Model Calibration**
- Calibration was performed **globally across all classes** and **individually per class** to ensure probability scores better reflected real-world confidence levels (visu_conf.py for samples visualization, IOU_hist.py for averaged statistics on the whole dataset. The dedicated dic should be computed first with IOU_csv.py).

## 📊 Results & Observations
- **Performance improved** with pretraining and augmentation.
- Errors were **biased towards the right side**, leading to additional post-processing steps.
- Self-supervised learning **did not yield significant gains** due to the difficulty of filtering unreliable pseudo-labels.
- Best result was attained with B3, pretraining on ImageNet, data augmentation and repatching, small post-processing (eliminating artefacts) : 67,3% IOU.

Code by:
L.R; 
O.H.

