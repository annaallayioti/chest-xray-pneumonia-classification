# Chest X-Ray Pneumonia Classification

A machine learning and deep learning project for **pneumonia detection from chest X-ray images**, comparing classical feature-based methods with convolutional neural networks and transfer learning.

The project explores both **binary classification** (Normal vs. Pneumonia) and an extended **three-class classification** task distinguishing Normal, Bacterial Pneumonia, and Viral Pneumonia.

> MSc coursework — Image Analysis and Processing, National and Kapodistrian University of Athens, 2026.

---

## Overview

Pneumonia diagnosis from chest X-ray images is an important medical image classification problem. This project investigates different machine learning approaches for automatically distinguishing healthy chest X-rays from pneumonia cases.

The complete pipeline includes:

- dataset exploration and class distribution analysis,
- image preprocessing and contrast enhancement,
- handcrafted feature extraction using **HOG** and **GLCM**,
- classical machine learning with **Random Forest** and **Linear SVM**,
- a custom **Convolutional Neural Network (CNN)**,
- transfer learning with **EfficientNetB0**,
- fine-tuning of the pretrained network,
- and an additional three-class pneumonia classification experiment.

---

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

The original dataset contains **5,856 chest X-ray images**:

| Split | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

The original validation set contains only 16 images, making it too small for reliable model evaluation.

To address this, the original training and validation sets were combined and a new **stratified 85/15 split** was created:

- **Training:** 4,447 images
- **Validation:** 785 images
- **Testing:** 624 images

The dataset also contains information that allows pneumonia cases to be separated into **Bacterial Pneumonia** and **Viral Pneumonia**, enabling an additional three-class experiment.

---

## Image Preprocessing

Chest X-ray images are converted to grayscale and resized to **224 × 224 pixels**.

The preprocessing pipeline includes:

- **Gaussian filtering** for noise reduction
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for local contrast enhancement
- image normalization
- conservative data augmentation for deep learning models

These operations improve local contrast and make anatomical structures more distinguishable while preserving the overall characteristics of the X-ray images.

---

## Classical Machine Learning

Two handcrafted feature representations are extracted from the preprocessed images:

### Histogram of Oriented Gradients (HOG)

HOG captures structural information such as edges, contours, and local gradient orientations.

### Gray-Level Co-occurrence Matrix (GLCM)

GLCM captures texture information using statistical properties including contrast, dissimilarity, homogeneity, energy, correlation, and ASM.

The combined representation produces **26,248 features per image**.

Two classifiers are evaluated:

- **Random Forest**
- **Linear Support Vector Machine (SVM)**

Feature standardization and class weighting are used where appropriate to improve training and account for class imbalance.

---

## Deep Learning

### Custom CNN

A custom Convolutional Neural Network is trained directly on the chest X-ray images.

The model learns hierarchical image representations automatically and serves as a deep learning baseline against the handcrafted HOG + GLCM approaches.

### EfficientNetB0

Transfer learning is performed using **EfficientNetB0** pretrained on ImageNet.

Training is performed in two stages:

1. **Feature extraction** with the EfficientNetB0 backbone frozen
2. **Fine-tuning** of the upper layers using a lower learning rate

This allows the network to first leverage pretrained visual representations and then adapt them more specifically to chest X-ray classification.

---

## Results

### Binary Classification

| Model | Accuracy | F1 Normal | F1 Pneumonia | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest | 70.03% | 0.36 | 0.80 | 0.58 |
| Linear SVM | 73.88% | 0.48 | 0.83 | 0.65 |
| Custom CNN | 78.85% | 0.68 | 0.84 | 0.76 |
| EfficientNetB0 — Frozen | 85.74% | 0.79 | 0.89 | 0.84 |
| **EfficientNetB0 — Fine-Tuned** | **87.18%** | **0.80** | **0.90** | **0.85** |

The results show a clear improvement as the models move from handcrafted features toward learned image representations.

The **fine-tuned EfficientNetB0 achieved the best overall performance**, reaching **87.18% test accuracy** and a **macro F1-score of 0.85**.

For pneumonia detection specifically, the fine-tuned model achieved a **recall of 0.97**, correctly identifying the large majority of pneumonia cases in the test set.

---

## Three-Class Classification

An additional experiment extends the problem to:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia

A fine-tuned EfficientNetB0 model achieved:

**Test Accuracy: 80.61%**

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bacterial Pneumonia | 0.80 | 0.87 | 0.84 |
| Normal | 0.82 | 0.82 | 0.82 |
| Viral Pneumonia | 0.78 | 0.68 | 0.73 |

The results indicate that the model can distinguish pneumonia subtypes to a meaningful extent, although **Viral Pneumonia remains the most challenging class**.

---

## Key Findings

- Classical models based on HOG and GLCM features provide a useful baseline but struggle particularly with Normal X-rays.
- The custom CNN improves substantially over handcrafted feature approaches.
- Transfer learning with EfficientNetB0 produces the strongest results.
- Fine-tuning further improves the pretrained model, achieving the highest binary classification accuracy and macro F1-score.
- The three-class experiment demonstrates that learned image representations can also capture differences between bacterial and viral pneumonia, although subtype classification remains more difficult than binary pneumonia detection.

---

## Technologies

- Python
- NumPy
- Pandas
- OpenCV
- scikit-image
- scikit-learn
- TensorFlow / Keras
- EfficientNetB0
- Matplotlib

---

## Repository Structure

```text
chest-xray-pneumonia-classification/
│
├── chest_xray_pneumonia_classification.ipynb
├── README.md
├── .gitignore
└── LICENSE
```

The dataset is downloaded directly from Kaggle and is therefore not included in the repository.

---

## Running the Notebook

The notebook can be executed in **Google Colab** or another Python environment with the required dependencies installed.

Kaggle API credentials must be configured before downloading the dataset.

The dataset is downloaded using:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

After extraction, the notebook performs dataset preparation, preprocessing, feature extraction, model training, and evaluation.

---
## Author

**Anna Allagioti**  


## License

This project is licensed under the MIT License.
