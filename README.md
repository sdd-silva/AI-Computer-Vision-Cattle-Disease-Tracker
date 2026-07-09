# AI-Computer-Vision-Cattle-Disease-Tracker


An end-to-end computer vision application for detecting cattle and classifying common livestock diseases using deep learning.

The system combines **YOLO11** for real-time cattle detection with a **ResNet18** image classifier trained using **PyTorch** to identify diseases from live camera footage or images.

---

## Features

- Real-time cattle detection using YOLO11
- Automatic cropping of detected animals
- Disease classification using a trained ResNet18 model
- Live webcam inference
- Prediction smoothing for stable real-time predictions
- Transfer learning using pretrained ImageNet weights
- Easily expandable to support additional diseases

---

## Supported Classes

The current model classifies cattle into:

- Healthy
- Lumpy Skin Disease (LSD)
- Foot-and-Mouth Disease (FMD)

---

## Project Architecture

```
Camera / Image
        │
        ▼
+----------------+
|     YOLO11     |
|Animal Detection|
+----------------+
        │
        ▼
Crop Detected Cow
        │
        ▼
+------------------+
|     ResNet18     |
|Disease Classifier|
+------------------+
        │
        ▼
Disease Prediction
        │
        ▼
Prediction Smoothing
        │
        ▼
Display Result
```

---

## Project Structure

```
animal-health-ai/
│
├── dataset/
│   ├── healthy/
│   ├── lumpy/
│   └── foot-and-mouth/
│
├── train.py
├── predict.py
├── camera.py
├── animal_camera.py
│
├── best_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- YOLO11 (Ultralytics)
- Pillow
- NumPy

---

## Model Training

The model uses transfer learning with a pretrained ResNet18 backbone.

Training pipeline:

- Image resizing (224 × 224)
- Data augmentation
- Train/Validation split
- CrossEntropyLoss
- Adam Optimizer
- Model checkpoint saving

Run:

```bash
python train.py
```

This generates:

```
best_model.pth
```

---

## Image Prediction

Predict a disease from a single image:

```bash
python predict.py
```

---

## Live Camera Detection

Run the complete real-time pipeline:

```bash
python animal_camera.py
```

The application will:

- Detect cattle
- Crop the detected animal
- Classify the disease
- Display confidence scores
- Smooth predictions across frames

---

## Dataset

The model was trained on a cattle disease dataset containing three classes:

- Healthy
- Lumpy Skin Disease
- Foot-and-Mouth Disease

Additional diseases can be added by expanding the dataset and retraining the model.

The dataset is too large to be included in this repository.

Download it from:

- Kaggle: https://www.kaggle.com/datasets/devang03mgr/cattle-diseases-datasets

---

## Future Improvements

- Animal tracking across frames
- Lesion localization using segmentation
- Multi-animal support
- Integration with IoT sensors
- GPS tracking
- LoRa communication
- Mobile dashboard
- Raspberry Pi / NVIDIA Jetson deployment
- Cloud-based monitoring

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Thefiesta223/AI-Computer-Vision-Cattle-Disease-Tracker.git
```

Install dependencies:

```bash
pip install -r "requirements.txt"
```

Run:

```bash
python animal_camera.py
```

---

## Model File

The trained model (`best_model.pth`) is not included in this repository due to GitHub file size limitations.

To generate the model:

```bash
python train.py
```

---

## License

This project is released under the MIT License.

---

## Author

Developed as an AI Computer Vision project for livestock disease detection using deep learning.
