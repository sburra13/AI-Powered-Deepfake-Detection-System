"""
app.py — DeepShield FastAPI Backend
=====================================
Endpoints:
  GET  /           → health check
  POST /predict    → deepfake detection on a single image
  POST /predict-video → deepfake detection on a video (frame sampling)

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import DeepfakeModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepShield API",
    description="AI-powered deepfake detection for images and videos.",
    version="1.0.0",
)

# Allow the HTML frontend (served from any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Model (loaded once at startup) ────────────────────────────────────────────
model = DeepfakeModel()
log.info("Model loaded and ready.")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"}
MAX_IMAGE_SIZE = 50  * 1024 * 1024   # 50 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024   # 200 MB

VIDEO_FRAME_SAMPLE = 16              # number of frames to sample from the video


# ── Helpers ───────────────────────────────────────────────────────────────────
async def save_upload(file: UploadFile, max_bytes: int) -> Path:
    """Stream upload to disk, reject if too large, return a unique path."""
    ext      = Path(file.filename).suffix
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"

    size = 0
    with open(tmp_path, "wb") as f:
        while chunk := await file.read(1024 * 64):   # 64 KB chunks
            size += len(chunk)
            if size > max_bytes:
                f.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large.")
            f.write(chunk)

    return tmp_path


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "message": "DeepShield API running 🛡"}


@app.post("/predict", tags=["Detection"])
async def predict_image(file: UploadFile = File(...)):
    """
    Detect whether an uploaded image is REAL or FAKE.

    Returns:
        prediction  : "REAL" | "FAKE"
        confidence  : float 0–100
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported image type: {file.content_type}")

    tmp_path = await save_upload(file, MAX_IMAGE_SIZE)

    try:
        label, confidence = model.predict(str(tmp_path))
        log.info("Image prediction: %s (%.2f%%)", label, confidence * 100)
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
        }

    except Exception as e:
        log.exception("Prediction failed: %s", e)
        raise HTTPException(500, f"Prediction error: {e}") from e

    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/predict-video", tags=["Detection"])
async def predict_video(file: UploadFile = File(...)):
    """
    Detect whether an uploaded video contains deepfake footage.

    Samples VIDEO_FRAME_SAMPLE frames evenly from the video, runs image
    prediction on each, and returns an aggregated verdict plus per-frame scores.

    Returns:
        prediction    : "REAL" | "FAKE"
        confidence    : float 0–100  (mean fake-class probability)
        frames_analysed: int
        frame_scores  : list[float]  (fake probability per sampled frame)
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(415, f"Unsupported video type: {file.content_type}")

    tmp_path = await save_upload(file, MAX_VIDEO_SIZE)

    try:
        frame_scores = _analyse_video_frames(str(tmp_path))

        if not frame_scores:
            raise HTTPException(422, "No face detected in any video frame.")

        mean_fake_prob = sum(frame_scores) / len(frame_scores)

        # Majority-vote label
        fake_frames = sum(1 for s in frame_scores if s > 0.5)
        label       = "FAKE" if fake_frames > len(frame_scores) / 2 else "REAL"

        # Confidence = probability of the winning class
        confidence = mean_fake_prob if label == "FAKE" else 1.0 - mean_fake_prob

        log.info(
            "Video prediction: %s (%.2f%%) across %d frames",
            label, confidence * 100, len(frame_scores)
        )

        return {
            "prediction":      label,
            "confidence":      round(confidence * 100, 2),
            "frames_analysed": len(frame_scores),
            "frame_scores":    [round(s, 4) for s in frame_scores],
        }

    except HTTPException:
        raise

    except Exception as e:
        log.exception("Video prediction failed: %s", e)
        raise HTTPException(500, f"Video prediction error: {e}") from e

    finally:
        tmp_path.unlink(missing_ok=True)


# ── Internal: video frame sampling ────────────────────────────────────────────
def _analyse_video_frames(video_path: str) -> list[float]:
    """
    Sample VIDEO_FRAME_SAMPLE frames evenly from the video.
    Returns a list of fake-class probabilities (one per frame that has a face).
    """
    import torch
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has no readable frames.")

    indices = [
        int(i * total_frames / VIDEO_FRAME_SAMPLE)
        for i in range(VIDEO_FRAME_SAMPLE)
    ]

    fake_probs = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Save frame as temp image, reuse existing model.predict()
        frame_path = str(UPLOAD_DIR / f"{uuid.uuid4()}_frame.jpg")
        cv2.imwrite(frame_path, frame)

        try:
            _, confidence_score, fake_prob = model.predict_with_prob(frame_path)
            fake_probs.append(fake_prob)
        except Exception as e:
            log.warning("Frame %d skipped: %s", idx, e)
        finally:
            Path(frame_path).unlink(missing_ok=True)

    cap.release()
    return fake_probs