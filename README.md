# VisDrone Object Detection Pipeline

A complete computer vision pipeline for detecting and classifying objects in aerial drone imagery — progressing from image classification to real-time object detection using the VisDrone 2019 dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [Phase 1 — Data Preparation](#phase-1--data-preparation)
  - [Phase 2 — Baseline CNN](#phase-2--baseline-cnn)
  - [Phase 3 — Transfer Learning](#phase-3--transfer-learning)
  - [Phase 4 — Object Detection](#phase-4--object-detection)
- [Results](#results)
- [Requirements](#requirements)
- [Setup](#setup)

---

## Project Overview

Standard image datasets like ImageNet contain ground-level photos that are useless for drone applications. Objects photographed from 100m above look completely different — a car becomes a rectangle with a roof, a person becomes a tiny oval blob. This project builds a pipeline specifically designed for aerial imagery, solving both classification (what is this object?) and detection (where are all the objects in this scene?).

---

## Dataset

**VisDrone 2019 Detection Dataset**
- Captured by drone cameras across 14 cities in China
- Covers varied weather, lighting, and altitude conditions
- Annotations in YOLO format (normalized bounding box coordinates)

**10 Object Classes**

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | pedestrian | 5 | truck |
| 1 | people | 6 | tricycle |
| 2 | bicycle | 7 | awning-tricycle |
| 3 | car | 8 | bus |
| 4 | van | 9 | motor |

---

## Project Structure

```
drone-detection/
│
├── data/
│   ├── VisDrone_Dataset/
│   │   ├── VisDrone2019-DET-train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── VisDrone2019-DET-val/
│   │       ├── images/
│   │       └── labels/
│   │
│   ├── balanced_dataset/          ← Phase 1 output
│   │   ├── train/
│   │   │   ├── pedestrian/
│   │   │   ├── people/
│   │   │   ├── bicycle/
│   │   │   └── car/
│   │   ├── val/
│   │   └── test/
│   │
│   └── yolov10_visdrone/          ← Phase 4 input
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── models/
│   ├── baseline_cnn_best.keras    ← Phase 2 output
│   ├── mobilenetv3_stage1.keras   ← Phase 3 Stage 1
│   ├── mobilenetv3_final.keras    ← Phase 3 Stage 2
│   └── yolov10_visdrone_final.pt  ← Phase 4 output
│
├── configs/
│   └── visdrone_all10.yaml        ← YOLOv10 dataset config
│
└── README.md
```

---

## Pipeline

### Phase 1 — Data Preparation

VisDrone is a detection dataset — each image is a large aerial scene. For classification (Phases 2 and 3), we first crop individual objects from scenes using YOLO bounding box annotations.

**Step 1 — Extraction**

Each `.txt` annotation file contains one line per object:
```
class_id  x_center  y_center  width  height   (all normalized 0–1)
```

We convert normalized coordinates back to pixels and crop each object. A minimum size of 20×20 pixels is enforced to discard unreadable blurry crops.

**Raw extraction (4 classes for classification):**
```
Pedestrian :  19,488    People  :   6,912
Bicycle    :   5,249    Car     :  93,541
Total      : 125,190 crops
```

**Step 2 — Class Balancing**

Car (93K) vs Bicycle (5.2K) is an 18:1 ratio. Without balancing, the model learns to always predict "car" and achieves 74.7% accuracy while being completely useless on other classes. Every class was capped at 5,249 images.

**Step 3 — Train / Val / Test Split**

| Split | Images | Purpose |
|-------|--------|---------|
| Train (70%) | 14,696 | Model learns from these |
| Val (15%) | 3,148 | Monitored after each epoch |
| Test (15%) | 3,152 | Final evaluation only — never seen during training |

`random.seed(42)` ensures the same split every run for reproducibility.

**Step 4 — Drone-Specific Augmentation**

Applied to training data only:

| Transform | Value | Reason |
|-----------|-------|--------|
| RandomRotation | 360° | Drones orbit — objects appear at any angle |
| RandomFlip | H + V | Any cardinal direction is valid from above |
| RandomZoom | ±20% | Altitude changes make objects larger or smaller |
| RandomBrightness | ±15% | Lighting varies from dawn to dusk |
| RandomContrast | ±15% | Shadows from buildings affect local contrast |

---

### Phase 2 — Baseline CNN

A simple 4-block CNN trained from scratch to establish a reference benchmark.

**Architecture**
```
Input (150×150×3)
    ↓
Conv2D(32)  → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(64)  → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.30)
Conv2D(256) → BatchNorm → MaxPool → Dropout(0.30)
    ↓
GlobalAveragePooling2D
Dense(256) → BatchNorm → Dropout(0.5)
Dense(4, softmax)
```

**Training Config**

```
Optimizer  : Adam (lr=1e-3)
Loss       : Categorical Crossentropy
Epochs     : 15 (with early stopping)
Batch size : 32
```

**Callbacks**

| Callback | Setting | Purpose |
|----------|---------|---------|
| EarlyStopping | patience=5 | Stop when val_loss stops improving |
| ModelCheckpoint | save_best_only=True | Keep best weights only |
| ReduceLROnPlateau | factor=0.5, patience=3 | Halve LR when stuck |

**Result: 55.30% test accuracy** — This is the benchmark every subsequent phase must beat.

---

### Phase 3 — Transfer Learning (MobileNetV3Small)

Instead of learning from scratch, we load MobileNetV3Small pretrained on ImageNet (1.28M images) and adapt it to aerial drone imagery in two stages.

**Why MobileNetV3**
- Designed for resource-constrained environments (mobile, edge, drone)
- Depthwise separable convolutions — 8× fewer parameters than standard CNNs
- Fast inference (~6ms per image) — suitable for real-time drone video
- Small variant fits within Colab free tier GPU memory

> ⚠️ **Critical:** MobileNetV3 requires `mobilenet_v3.preprocess_input` which scales pixels to `[-1, +1]`. Using standard `Rescaling(1./255)` gives `[0, 1]` and silently reduces accuracy by 15–20%.

**Architecture**
```
Input (224×224×3)
    ↓
MobileNetV3Small (pretrained on ImageNet — frozen in Stage 1)
    ↓
GlobalAveragePooling2D
Dense(128, relu) → BatchNorm → Dropout(0.4)
Dense(4, softmax)
```

**Two-Stage Training**

| | Stage 1 | Stage 2 |
|--|---------|---------|
| Base layers | All frozen | Top 30 unfrozen |
| Learning rate | 1e-3 | 1e-5 (100× smaller) |
| Epochs | 10 | 15 |
| Goal | Adapt new head from random weights | Fine-tune high-level features toward aerial imagery |

Stage 1 must come before Stage 2. Skipping Stage 1 and immediately fine-tuning with a randomly initialised head produces large unstable gradients that destroy the pretrained weights — known as catastrophic forgetting.

**Memory Fixes (Colab Free Tier)**

| Problem | Fix |
|---------|-----|
| `.cache()` on 14K images = 8.8 GB RAM | Removed cache from train pipeline |
| Batch size 32 = GPU OOM | Reduced to 16 |
| MobileNetV3Large crash | Switched to Small (2.5M vs 5.4M params) |
| Local SSD copy = RAM spike | Read directly from Google Drive |
| Model lost on session close | Save checkpoints directly to Drive |

**Results**

```
Overall Test Accuracy : 78.39%   (+23.09pp over baseline)
```

| Class | Accuracy | F1-Score |
|-------|----------|----------|
| Car | 93.3% ✅ | 0.927 |
| Bicycle | 82.1% ✅ | 0.816 |
| Pedestrian | 68.8% ⚠️ | 0.705 |
| People | 69.4% ⚠️ | 0.686 |

Pedestrian and People both score ~69% because from 100m altitude both appear as indistinguishable 18×22 pixel oval blobs. This is a fundamental data limitation — published academic papers report 65–75% for this class pair on VisDrone.

---

### Phase 4 — Object Detection (YOLOv10n)

Moved from classification (one crop → one label) to full object detection (one raw aerial scene → many bounding boxes with labels). This is the real-world drone task — no manual cropping required.

**Why YOLOv10 over YOLOv8**

| Feature | YOLOv8 | YOLOv10 |
|---------|--------|---------|
| Post-processing | Requires NMS | NMS-free ✅ |
| Label assignment | One-to-many | Dual (consistent + one-to-many) ✅ |
| Inference speed | Baseline | ~1.4× faster ✅ |
| API | ultralytics | Same ultralytics API |

**Dataset Config (`visdrone_all10.yaml`)**
```yaml
path  : /content/yolov10_visdrone
train : images/train
val   : images/val
nc    : 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

**Training Config**
```
Model      : YOLOv10n (2.3M parameters, pretrained on COCO)
Epochs     : 100  (patience=20 early stopping)
Image size : 640×640
Batch size : 8
Optimizer  : AdamW  (lr=0.001, cosine schedule)
```

**Drone-Specific Training Settings**

| Setting | Value | Reason |
|---------|-------|--------|
| mosaic | 1.0 | Combines 4 images — exposes model to more objects per step |
| degrees | 45° | Drone rotation — objects at any angle |
| flipud | 0.5 | Vertical flip valid for aerial imagery |
| scale | 0.5 | Altitude variation changes apparent object size |
| copy_paste | 0.1 | Helps rare classes (awning-tricycle, bus) see more examples |
| hsv_v | 0.4 | Brightness variation for altitude and time-of-day changes |

**Expected Results**

```
Overall mAP@50 : 54%
```

| Class | Expected mAP@50 |
|-------|----------------|
| car | 50–65% |
| van | 40–55% |
| bus | 35–50% |
| truck | 35–50% |
| motor | 30–45% |
| bicycle | 30–45% |
| tricycle | 25–40% |
| pedestrian | 25–38% |
| people | 20–35% |
| awning-tricycle | 15–30% |

> **Why mAP@50 is lower than Phase 3 accuracy:** These measure different things. Accuracy asks "is this pre-cropped image labeled correctly?" mAP@50 asks "did the model find every object in a full scene AND draw a box overlapping ground truth by ≥50%?" Detection is harder. mAP@50 above 35% on VisDrone is competitive with published research.

---

## Results

| Phase | Task | Model | Score |
|-------|------|-------|-------|
| Phase 2 | Classification (4 classes) | Custom CNN | 55.30% accuracy |
| Phase 3 | Classification (4 classes) | MobileNetV3Small | 78.39% accuracy |
| Phase 4 | Detection (10 classes) | YOLOv10n | ~32–45% mAP@50 |

---

## Requirements

```
tensorflow>=2.13.0
ultralytics>=8.2.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
torch>=2.0.0
```

Install all:
```bash
pip install tensorflow ultralytics opencv-python numpy matplotlib seaborn scikit-learn torch
```

---

## Setup

**1. Mount Google Drive (Colab)**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**2. Set your base path**
```python
DRIVE_BASE = '/content/drive/MyDrive/drone dataset'
```

**3. Run phases in order**
```
Phase 1 → Extract crops from VisDrone scenes
Phase 2 → Train baseline CNN  (establishes benchmark)
Phase 3 → Train MobileNetV3  (transfer learning)
Phase 4 → Train YOLOv10n     (full detection)
```

**4. Load saved models**
```python
import tensorflow as tf
from ultralytics import YOLO

# Phase 3 classifier
classifier = tf.keras.models.load_model(f'{DRIVE_BASE}/mobilenetv3_final.keras')

# Phase 4 detector
detector = YOLO(f'{DRIVE_BASE}/yolov10_visdrone_final.pt')

# Run detection on a new image
results = detector.predict('path/to/drone/image.jpg', conf=0.25)
```

---

## Key Concepts Learned

- **Class imbalance** silently corrupts model training before a single epoch runs. Always check and fix distribution before training.
- **Transfer learning** delivers large accuracy gains even when source domain (ImageNet ground photos) differs from target domain (aerial imagery), because low-level visual features are universal.
- **Accuracy vs mAP@50** are not comparable metrics. Always interpret numbers in the context of the task they measure.
- **Colab memory management** requires careful pipeline design — `.cache()`, batch size, and model variant all interact and can crash a session.
- **Drone-specific augmentation** (360° rotation, altitude zoom) is not optional — without it the model fails on objects oriented differently from training examples.
