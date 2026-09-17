# Breast Ultrasound Image Segmentation & Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-Segmentation%20%2B%20Classification-purple)

A multi-task deep learning system that simultaneously performs **tumour segmentation** and **malignancy classification** on breast ultrasound images using a shared U-Net encoder architecture.

---

## 🔬 Overview

Accurate breast cancer diagnosis requires two complementary steps: localising the tumour region (segmentation) and determining its nature — benign or malignant (classification). This project tackles both tasks in a single end-to-end model that shares a convolutional backbone between a pixel-wise segmentation decoder and an image-level classification head, improving efficiency and allowing the two tasks to reinforce each other during training.

---

## 📊 Dataset

**[BUSI — Breast Ultrasound Images Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)**

| Category | Description |
|----------|-------------|
| **Normal** | Healthy breast tissue (no tumour) |
| **Benign** | Non-cancerous tumour |
| **Malignant** | Cancerous tumour |

Each sample includes:
- A grayscale ultrasound image
- A corresponding binary segmentation mask (tumour region)

> **Citation:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images.* Data in Brief. 2020;28:104863. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863)

---

## 🏗️ Model Architecture

The model uses a **U-Net backbone** whose encoder is shared between both task branches:

```
Input Ultrasound Image
          │
    ┌─────▼─────┐
    │  Encoder  │   ← Shared convolutional feature extractor
    │ (U-Net)   │     (Conv blocks + MaxPooling)
    └─────┬─────┘
          │
    ┌─────┴──────────────────────┐
    ▼                            ▼
┌──────────────────┐    ┌──────────────────────┐
│  Segmentation    │    │   Classification     │
│    Decoder       │    │       Head           │
│                  │    │                      │
│  U-Net skip      │    │  GlobalAvgPool →     │
│  connections +   │    │  Dense(256) →        │
│  UpSampling →    │    │  Dropout →           │
│  sigmoid output  │    │  Dense(3, softmax)   │
│  (pixel mask)    │    │  (Normal/Benign/Mal) │
└──────────────────┘    └──────────────────────┘
```

---

## 📉 Loss Functions

| Task | Loss Function |
|------|--------------|
| Segmentation | Binary Cross-Entropy + Dice Loss |
| Classification | Categorical Cross-Entropy |
| **Combined** | Weighted sum of segmentation + classification loss |

---

## 🏋️ Training Strategy

- **Optimiser:** Adam
- **Learning rate scheduling:** ReduceLROnPlateau — reduces LR when validation loss plateaus
- **Early stopping:** Halts training when validation loss stops improving
- **Data augmentation:**
  - Horizontal and vertical flips
  - Random rotation
  - Brightness and contrast adjustments

---

## 📏 Evaluation Metrics

**Segmentation (per image)**

| Metric | Description |
|--------|-------------|
| Dice Coefficient | Overlap between predicted and ground-truth mask |
| IoU | Intersection over Union |
| Pixel Accuracy | % of pixels correctly classified |

**Classification (per image)**

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct class predictions |
| Precision / Recall / F1 | Per-class breakdown |
| Confusion Matrix | Full misclassification analysis |

---

## 📁 Project Structure

```
Breast-Ultrasound-Image-Segmentation-and-Classification/
├── model/
│   ├── unet_multitask.py       # Model architecture definition
│   └── losses.py               # Custom loss functions (Dice + BCE)
├── data/
│   └── preprocessing.py        # Data loading, augmentation, mask processing
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation + metric reporting
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Abtahi0712/Breast-Ultrasound-Image-Segmentation-Classification.git
cd Breast-Ultrasound-Image-Segmentation-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the BUSI dataset from the [official source](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and place the images and masks in the `data/` directory, maintaining the folder structure:
```
data/
├── normal/
├── benign/
└── malignant/
```

### 4. Train the model
```bash
python train.py
```

### 5. Evaluate
```bash
python evaluate.py
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Model building and training |
| `NumPy` | Array operations |
| `OpenCV` | Image loading and preprocessing |
| `Matplotlib / Seaborn` | Visualisation |
| `scikit-learn` | Metrics (confusion matrix, classification report) |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
