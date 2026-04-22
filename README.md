# AI-Powered-Deepfake-Detection-System-
Detect AI-generated and manipulated facial images and videos in real time using deep learning.

## About
DeepFake Detection is a full-stack AI application that identifies whether an image or video contains a synthetically generated or manipulated human face. Built as a Final Year Project, it combines a fine-tuned Xception convolutional neural network with a FastAPI backend and a responsive HTML/CSS/JavaScript frontend — making state-of-the-art deepfake detection accessible to non-technical users through a clean web interface.
As deepfake technology becomes increasingly accessible, the ability to verify the authenticity of visual media is critical for journalists, researchers, and the general public. This project addresses that need with a fast, accurate, and deployable detection tool.

## Features
Image Detection — Upload any portrait photo and receive a REAL or FAKE verdict with a confidence score in under 500ms
Video Detection — Upload a video clip for frame-by-frame analysis with per-frame confidence breakdown and overall verdict
Face-Focused Analysis — Haar cascade face detection isolates the facial region before classification, improving accuracy
Real-Time Results — Animated confidence bar, natural-language explanation, and instant visual feedback
Privacy First — Uploaded files are permanently deleted immediately after prediction — nothing is stored
Drag and Drop — Intuitive upload interface with live preview for both images and videos
REST API — Full FastAPI backend with documented endpoints, usable independently of the frontend

## Tech Stack
Layer            Technology  
Model            Xception CNN (fine-tuned, pretrained on ImageNet)
Training         PyTorch 2.5 + CUDA, albumentations, timm
Backend          FastAPI, Uvicorn, OpenCV
Frontend         HTML5, CSS3, Vanilla JavaScript
Face Detection   OpenCV Haar Cascade
Evaluation       scikit-learn (AUC, ROC, confusion matrix)

## Project Structure
DeepFake/
├── index.html              # Home page
├── detect-image.html       # Image detection page
├── detect-video.html       # Video detection page
├── styles.css              # Shared stylesheet
├── app.py                  # FastAPI backend
├── model.py                # Xception inference module
├── train.py                # Model training script
├── prepare_dataset.py      # Dataset preparation (merge images + video frames)
├── model.pth               # Trained model weights
├── image_dataset/          # Raw image dataset
│   ├── real/
│   └── fake/
├── video_dataset/          # Raw video dataset
│   ├── real/
│   └── fake/
└── dataset/                # Merged training dataset (auto-generated)
    └── train/
        ├── real/
        └── fake/


## Getting Started
Prerequisites
-Python 3.12 (required for CUDA support)
-NVIDIA GPU with CUDA 12.1 (recommended)

## Installation
### Clone the repository
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection

### Install dependencies
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
py -3.12 -m pip install timm albumentations opencv-python scikit-learn fastapi uvicorn python-multipart

## Usage
Step 1 — Prepare dataset (merge image + video datasets)
py -3.12 prepare_dataset.py

Step 2 — Train the model
py -3.12 train.py --epochs 15

Step 3 — Start the API
py -3.12 -m uvicorn app:app --host 0.0.0.0 --port 8000

Step 4 — Serve the frontend
py -3.12 -m http.server 3000
Then open http://localhost:3000 in your browser.

## API Endpoints
Method        Endpoint      Description
GET            /            Health check
POST          /predict      Detect deepfake in an image (max 50MB)
POST      /predict-video    Detect deepfake in a video (max 200MB)

## Example
curl -X POST http://localhost:8000/predict \
  -F "file=@face.jpg"

  {
  "prediction": "FAKE",
  "confidence": 94.7
}

## How It Works?
-User uploads an image or video through the web interface
-The file is sent to the FastAPI backend via a multipart POST request
-OpenCV's Haar cascade detects and crops the largest face region
-The face crop is normalised and passed through the fine-tuned Xception model
-A softmax confidence score is returned — REAL or FAKE
-The uploaded file is immediately deleted from the server
-The result is displayed with an animated confidence bar

## Acknowledgements

FaceForensics++ — deepfake dataset
timm — Xception pretrained weights
albumentations — image augmentation pipeline
Chollet, F. (2017) — Xception architecture

## License
This project is for academic and educational purposes.



