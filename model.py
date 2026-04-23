"""
model.py — DeepShield Inference Module
========================================
Loads the trained Xception model and exposes:

  model.predict(image_path)             → (label, confidence)
  model.predict_with_prob(image_path)   → (label, confidence, fake_prob)

Used by app.py for both image and video (frame-level) inference.
"""

import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
from albumentations.pytorch import ToTensorV2

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Class index mapping — must match the folder order used in ImageFolder during training.
# ImageFolder assigns labels alphabetically: fake=0, real=1
IDX_TO_LABEL = {0: "FAKE", 1: "REAL"}


class DeepfakeModel:
    def __init__(self, weights_path: str = "model.pth") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", self.device)

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. "
                "Run train.py first to generate model.pth."
            )

        self.model = timm.create_model("xception", pretrained=False, num_classes=2)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        log.info("Xception model loaded from '%s'.", weights_path)

        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ── Public API ─────────────────────────────────────────────────────────
    def predict(self, image_path: str) -> tuple[str, float]:
        """
        Returns (label, confidence) where:
          label      : "REAL" | "FAKE"
          confidence : float in [0, 1]
        """
        label, confidence, _ = self.predict_with_prob(image_path)
        return label, confidence

    def predict_with_prob(self, image_path: str) -> tuple[str, float, float]:
        """
        Returns (label, confidence, fake_prob) where:
          label      : "REAL" | "FAKE"
          confidence : probability of the predicted class, float in [0, 1]
          fake_prob  : raw probability assigned to the FAKE class
        """
        tensor = self._preprocess(image_path)          # shape: [1, 3, 224, 224]

        with torch.no_grad():
            logits = self.model(tensor)                # [1, 2]
            probs  = torch.softmax(logits, dim=1)[0]   # [2]

        fake_prob = float(probs[0].item())             # class 0 = FAKE
        real_prob = float(probs[1].item())             # class 1 = REAL

        if fake_prob >= real_prob:
            return "FAKE", fake_prob, fake_prob
        else:
            return "REAL", real_prob, fake_prob

    # ── Private helpers ─────────────────────────────────────────────────────
    def _preprocess(self, image_path: str) -> torch.Tensor:
        img = cv2.imread(image_path)
        if img is None:
            raise OSError(f"Cannot read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face      = self._extract_face(img)
        augmented = self.transform(image=face)
        return augmented["image"].unsqueeze(0).to(self.device)

    def _extract_face(self, img: np.ndarray) -> np.ndarray:
        """
        Return the largest detected face crop.
        Falls back to a centred square crop with clamped bounds.
        """
        gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return img[y : y + h, x : x + w]

        # Fallback: safe centred crop
        h, w   = img.shape[:2]
        cx, cy = w // 2, h // 2
        half   = min(w, h) // 2
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        return img[y0:y1, x0:x1]