"""
train.py — Final Clean Version (Python 3.12 + CUDA)
=====================================================
Run with:
    py -3.12 train.py --epochs 5
"""

import argparse
import logging
import sys
import os
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="dataset/train")
    p.add_argument("--output",      default="model.pth")
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=3e-5)
    p.add_argument("--val_split",   type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=0)
    return p.parse_args()


# ── Face Extraction ───────────────────────────────────────────────────────────
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_face(img: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    if H < 10 or W < 10:
        return cv2.resize(img, (224, 224))

    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.3,
                                       minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = img[y:y+h, x:x+w]
        if crop.size > 0:
            return crop

    # Safe centred fallback
    cx, cy = W // 2, H // 2
    half   = min(W, H) // 2
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)
    crop = img[y0:y1, x0:x1]
    return crop if crop.size > 0 else cv2.resize(img, (224, 224))


# ── Augmentation ─────────────────────────────────────────────────────────────
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

TRAIN_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ImageCompression(quality_range=(30, 100), p=0.5),
    A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    A.CoarseDropout(
        num_holes_range=(1, 8),
        hole_height_range=(8, 16),
        hole_width_range=(8, 16),
        p=0.3
    ),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

VAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, subset: Subset, transform: A.Compose) -> None:
        self.subset    = subset
        self._base     = subset.dataset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        global_idx  = self.subset.indices[idx]
        path, label = self._base.samples[global_idx]

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face = extract_face(img)
        return self.transform(image=face)["image"], label


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss, 100.0 * correct / total if total else 0.0


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct, total        = 0, 0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    val_acc = 100.0 * correct / total if total else 0.0
    auc     = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return val_acc, auc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # GPU is required — verified working with py -3.12
    device = torch.device("cuda")
    log.info("GPU        : %s", torch.cuda.get_device_name(0))
    log.info("PyTorch    : %s", torch.__version__)
    log.info("Epochs     : %d", args.epochs)
    log.info("Batch size : %d", args.batch_size)

    # ── Data ──────────────────────────────────────────────────────────────
    data_path = Path(args.data_dir)
    if not data_path.exists():
        log.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    real_count = len(list((data_path / "real").glob("*"))) if (data_path / "real").exists() else 0
    fake_count = len(list((data_path / "fake").glob("*"))) if (data_path / "fake").exists() else 0
    log.info("Dataset    : %d real | %d fake | %d total",
             real_count, fake_count, real_count + fake_count)

    base_dataset = datasets.ImageFolder(args.data_dir)
    log.info("Class mapping: %s", base_dataset.class_to_idx)

    val_size   = int(args.val_split * len(base_dataset))
    train_size = len(base_dataset) - val_size
    train_sub, val_sub = random_split(
        base_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    log.info("Split      : %d train | %d val", len(train_sub), len(val_sub))

    loader_kw = dict(
        batch_size         = args.batch_size,
        num_workers        = args.num_workers,
        pin_memory         = True,
        persistent_workers = False,
    )
    train_loader = DataLoader(FaceDataset(train_sub, TRAIN_TRANSFORM),
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(FaceDataset(val_sub, VAL_TRANSFORM),
                              shuffle=False, **loader_kw)

    # ── Class weights ─────────────────────────────────────────────────────
    labels  = [l for _, l in base_dataset.samples]
    counts  = Counter(labels)
    weights = torch.tensor(
        [1.0 / counts[i] for i in range(len(counts))]
    ).to(device)
    log.info("Class weights: %s",
             {k: round(weights[v].item(), 5)
              for k, v in base_dataset.class_to_idx.items()})

    # ── Model ─────────────────────────────────────────────────────────────
    model     = timm.create_model("xception", pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────
    best_auc = 0.0
    log.info("=" * 55)
    log.info("Training started")
    log.info("=" * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device)
        val_acc, auc = validate(model, val_loader, device)
        scheduler.step()

        log.info(
            "Epoch %02d/%02d | Loss=%.3f | Train=%.2f%% | Val=%.2f%% | AUC=%.4f",
            epoch, args.epochs, train_loss, train_acc, val_acc, auc,
        )

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.output)
            log.info("  [SAVED] Best AUC=%.4f -> %s", best_auc, args.output)

    log.info("=" * 55)
    log.info("Training complete. Best AUC: %.4f", best_auc)
    log.info("model.pth is ready.")
    log.info("Start API: py -3.12 -m uvicorn app:app --host 0.0.0.0 --port 8000")
    log.info("=" * 55)


if __name__ == "__main__":
    main()