# 🚗 Driver Drowsiness Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?logo=tensorflow)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A deep learning-based **driver drowsiness detection** system that classifies driver states as **Awake** or **Drowsy** from facial images. Three model architectures are trained and compared: **Custom CNN**, **MobileNetV2**, and **YOLOv8**.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Training Time |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|:-------------:|
| Custom CNN | 99.97% | 99.98% | 99.95% | 99.96% | 1.000 | ~105 min |
| MobileNetV2 | 99.95% | 99.97% | 99.94% | 99.96% | 1.000 | ~265 min |
| **YOLOv8** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** | ~27 min |

> Tested on 6,271 images (2,918 awake + 3,353 drowsy)

---

## 📁 Project Structure

```
driver-drowsiness-detection/
├── notebook/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb    # Data preprocessing & splitting
│   ├── 03_CNN.ipynb              # Custom CNN training
│   ├── 04_MobileNetV2.ipynb      # MobileNetV2 transfer learning
│   ├── 05_YOLOv8.ipynb           # YOLOv8 classification training
│   ├── 06_Evaluation.ipynb       # Model comparison & evaluation
│   └── 07_RealTime.ipynb         # Real-time webcam detection
├── results/
│   ├── cnn/                      # CNN metrics & plots
│   ├── mobilenetv2/              # MobileNetV2 metrics & plots
│   ├── yolo/                     # YOLOv8 metrics & plots
│   ├── comparison/               # Cross-model comparison charts
│   └── plots/                    # Dataset analysis plots
├── dataset/                      # ⚠️ Not included (see Dataset section)
├── models/                       # ⚠️ Not included (large files)
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧠 Models Overview

### 1. Custom CNN
A lightweight architecture trained from scratch.
- **Layers:** 4× Conv2D → GlobalAveragePooling → Dense
- **Input:** 128×128 RGB
- **Parameters:** ~421K
- **Best for:** Low-resource environments

### 2. MobileNetV2 (Transfer Learning)
Pre-trained on ImageNet, fine-tuned for drowsiness detection.
- **Training phases:** Feature extraction → Fine-tuning
- **Input:** 224×224 RGB
- **Best for:** Balance of accuracy and speed

### 3. YOLOv8 Classification
State-of-the-art classification using YOLOv8n-cls.
- **Base model:** `yolov8n-cls.pt`
- **Input:** 224×224 RGB
- **Best for:** Real-time inference with highest accuracy

---

## 🗂️ Dataset

The dataset contains face images labeled as **awake** or **drowsy**.

| Split | Awake | Drowsy | Total |
|-------|------:|------:|------:|
| Train | ~11,700 | ~13,500 | ~25,200 |
| Validation | ~2,500 | ~2,900 | ~5,400 |
| Test | 2,918 | 3,353 | **6,271** |

> The dataset is **not included** in this repository due to size constraints.  
> Please prepare your own dataset following the structure in `02_Preprocessing.ipynb`.

---

## ⚙️ Installation

### Requirements
- Python 3.9+
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/harieyudha/Driver-drowsiness-detection..git driver-drowsiness-detection
cd driver-drowsiness-detection

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Notebooks in Order

```bash
jupyter notebook
```

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_EDA.ipynb` | Explore and visualize the dataset |
| 2 | `02_Preprocessing.ipynb` | Prepare and split data |
| 3 | `03_CNN.ipynb` | Train custom CNN model |
| 4 | `04_MobileNetV2.ipynb` | Train MobileNetV2 model |
| 5 | `05_YOLOv8.ipynb` | Train YOLOv8 model |
| 6 | `06_Evaluation.ipynb` | Compare all models |
| 7 | `07_RealTime.ipynb` | Run real-time webcam detection |

---

## 📈 Results & Visualizations

All evaluation outputs are saved in `results/`:

- **Confusion matrices** per model
- **ROC curves** per model
- **Cross-model comparison** charts (bar, radar, bubble)
- **Training history** plots

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow 2.x / Keras |
| Object Detection | Ultralytics YOLOv8 |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## � License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [TensorFlow / Keras](https://tensorflow.org) for deep learning backbone
- Dataset contributors for making drowsiness detection research possible
