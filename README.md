# 🧠 Brain Tumor Detection System using Deep Learning (VGG16)

An advanced AI-powered Brain Tumor Detection System that leverages transfer learning with a fine-tuned VGG16 model to classify brain MRI scans into one of four categories: **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.

Achieving **98.2% accuracy** and **0.98 AUC-ROC**, this tool is designed to support radiologists with rapid, accurate, and explainable diagnostic predictions.

---

## 📌 Features

- ✅ **High Accuracy**: 98.2% classification accuracy on test data  
- ⚡ **Fast Inference**: MRI scan predictions in seconds  
- 🔍 **Explainable AI**: Heatmaps and confidence scores for transparency  
- 🌐 **Flask API**: Easy integration with healthcare platforms  
- 🧠 **Transfer Learning**: Fine-tuned VGG16 for optimal feature extraction  
- 📊 **Evaluation Tools**: Confusion matrix, ROC curve, AUC score  
- 🔄 **Robust Augmentation**: Handles diverse imaging conditions

---

## ⚙️ Methodology

### 1. Data Preparation
- Collected MRI scans for 4 tumor types (Glioma, Meningioma, Pituitary, No Tumor)
- Automated directory-based image loading
- Applied real-time shuffling to avoid ordering bias

### 2. Image Processing
- Standardized image size to 128×128  
- Real-time augmentations:  
  - Brightness adjustment (0.8–1.2×)  
  - Contrast variation (0.8–1.2×)  
  - Normalization to [0, 1]

### 3. Model Architecture
- **VGG16** pretrained on ImageNet  
  - Frozen convolutional base  
  - Unfroze last 3 layers for fine-tuning  
  - Custom classification head:  
    - `Flatten → Dense(128, ReLU) → Dropout(0.3) → Dense(4, Softmax)`

### 4. Training
- Optimizer: Adam (lr=0.0001)  
- Loss Function: Sparse Categorical Crossentropy  
- Batch Size: 20  
- Metrics: Accuracy, AUC

### 5. Evaluation
- Generated classification report  
- Confusion matrix, ROC curves, AUC scores  
- Visualized predictions with confidence values

---

## 🧪 Results

| Metric           | Value     |
|------------------|-----------|
| Accuracy         | 98.2%     |
| AUC-ROC Score    | 0.98      |
| Loss Function    | Sparse Categorical Crossentropy |

---

## 🚀 Installation & Usage

### Prerequisites

Install required packages:

```bash
pip install tensorflow flask opencv-python numpy matplotlib scikit-learn
