# YOLOv9 Dendritic Spine Detection 🧠

> Advanced computer vision model for automated detection of dendritic spines in microscopy images using YOLOv9


## 🎯 Project Overview

This project implements a state-of-the-art YOLOv9 object detection model to automatically identify and locate dendritic spines in neuronal microscopy images. Dendritic spines are critical structures in neuroscience research, and automated detection significantly accelerates analysis workflows.

**Key Achievements:**
- 📊 **96.4% mAP** on validation set
- 🎯 **94.1% Precision** and **91.0% Recall**
- 🔬 **1,836 images** processed and annotated
- ⚡ **5.3ms inference time** per image

## 🔬 Dataset & Preprocessing

**Original Dataset:**
- 1,836 spine microscopy images with bounding box annotations
- Custom coordinate format converted to YOLO format
- 80/20 train-validation split
- Image standardization to 416x416 pixels

**Data Pipeline:**
```
Raw Spine Images → Coordinate Conversion → YOLO Format → Training Ready
```

## 🏗️ Model Architecture

- **Base Model:** YOLOv9m (Medium)
- **Parameters:** 20,013,715 trainable parameters
- **Input Size:** 416x416 pixels
- **Classes:** 1 (dendritic_spine)
- **Training:** 100 epochs with early stopping

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| Precision | 94.1% |
| Recall | 91.0% |
| mAP@0.5 | 96.4% |
| Inference Speed | 5.3ms |
| Training Time | 0.909 hours |

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, Ultralytics YOLOv9
- **Computer Vision:** OpenCV, PIL
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Environment:** Google Colab with GPU acceleration

## 🚀 Quick Start

### Prerequisites
```bash
pip install ultralytics opencv-python numpy matplotlib pyyaml
```

### Training the Model
```python
# Install dependencies
!pip install ultralytics

# Train YOLOv9 model
!yolo train model=yolov9m.pt data=dendrites.yaml epochs=100 imgsz=416
```

### Running Inference
```python
# Load trained model
from ultralytics import YOLO
model = YOLO('path/to/best.pt')

# Run inference
results = model.predict('path/to/test/images')
```


```

## 🔄 Data Processing Pipeline

The project includes a comprehensive data preprocessing pipeline:

1. **Data Extraction:** Unzip and organize spine image dataset
2. **Format Conversion:** Convert bounding box coordinates to YOLO format
3. **Normalization:** Standardize coordinates to [0,1] range
4. **Dataset Split:** 80% training, 20% validation
5. **Quality Assurance:** Validate all annotations and image integrity

## 📊 Model Training Process

**Training Configuration:**
- Optimizer: AdamW with automatic mixed precision
- Learning Rate: Adaptive scheduling
- Batch Size: Optimized for GPU memory
- Data Augmentation: Built-in YOLO augmentations
- Loss Functions: Box loss, Classification loss, DFL loss

**Training Progression:**
- Epochs 1-50: Initial convergence
- Epochs 51-80: Fine-tuning and optimization  
- Epochs 81-100: Final refinement and validation

## 🎯 Results & Evaluation

### Confusion Matrix
The model achieves excellent classification performance with minimal false positives:
- **True Positives:** 1,416 dendritic spines correctly detected
- **False Negatives:** 77 spines missed
- **False Positive Rate:** < 5%

### Performance Visualization
Training curves show smooth convergence across all loss metrics:
- Box Loss: Steady decrease to 1.24
- Classification Loss: Converged to 0.87  
- DFL Loss: Optimized to 1.06



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

