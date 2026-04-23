"""
STEP 1 — Run this first: prepare_dataset.py
============================================
This script:
  1. Reads your existing image dataset
  2. Extracts frames from your video dataset
  3. Merges everything into one clean training folder: dataset/train/real/ and dataset/train/fake/
  4. Prints a summary so you can verify before training

USAGE:
    python prepare_dataset.py

EDIT ONLY THESE 4 PATHS BELOW:
"""

import cv2
import shutil
import os
from pathlib import Path

# ═══════════════════════════════════════════════════
# ✏️  EDIT THESE 4 PATHS TO MATCH YOUR FOLDER NAMES
# ═══════════════════════════════════════════════════

IMAGE_REAL_FOLDER = Path("image_dataset/real")   # your real photos folder
IMAGE_FAKE_FOLDER = Path("image_dataset/fake")   # your fake photos folder

VIDEO_REAL_FOLDER = Path("video_dataset/real")   # your real videos folder
VIDEO_FAKE_FOLDER = Path("video_dataset/fake")   # your fake videos folder

# ═══════════════════════════════════════════════════
# Output folder — do NOT change this
# ═══════════════════════════════════════════════════
OUTPUT_REAL = Path("dataset/train/real")
OUTPUT_FAKE = Path("dataset/train/fake")

# How many frames to extract per video (10 = good balance of diversity vs speed)
FRAMES_PER_VIDEO = 10

# Supported file types
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


# ───────────────────────────────────────────────────
def check_folder(path: Path, label: str):
    """Verify folder exists and has files."""
    if not path.exists():
        print(f"  ❌ NOT FOUND: {path}  ← fix the path for {label}")
        return False
    files = list(path.iterdir())
    print(f"  ✅ {path}  ({len(files)} files)")
    return True


def copy_images(src_folder: Path, dst_folder: Path, label: str):
    """Copy all images from src into dst."""
    dst_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for f in src_folder.iterdir():
        if f.suffix.lower() in IMAGE_EXTS:
            # Avoid overwriting — add prefix
            dst = dst_folder / f"img_{f.name}"
            shutil.copy2(f, dst)
            copied += 1
        else:
            skipped += 1
    print(f"  📸 {label}: copied {copied} images" +
          (f" (skipped {skipped} non-image files)" if skipped else ""))
    return copied


def extract_video_frames(src_folder: Path, dst_folder: Path, label: str):
    """Extract FRAMES_PER_VIDEO frames from each video in src_folder into dst_folder."""
    dst_folder.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    total_videos = 0
    failed = 0

    videos = [f for f in src_folder.iterdir() if f.suffix.lower() in VIDEO_EXTS]

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"    ⚠️  Cannot open: {video.name}")
            failed += 1
            continue

        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_in_video == 0:
            cap.release()
            failed += 1
            continue

        # Pick evenly spaced frame indices
        indices = [
            int(i * total_frames_in_video / FRAMES_PER_VIDEO)
            for i in range(FRAMES_PER_VIDEO)
        ]

        saved = 0
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            out_name = dst_folder / f"vid_{video.stem}_f{idx}.jpg"
            cv2.imwrite(str(out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        cap.release()
        total_frames += saved
        total_videos += 1

    print(f"  🎬 {label}: extracted {total_frames} frames from {total_videos} videos" +
          (f" (⚠️  {failed} videos failed)" if failed else ""))
    return total_frames


# ───────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  DEEPFAKE DETECTION — DATASET PREPARATION")
    print("="*55)

    # Step 1: Check all source folders exist
    print("\n[1] Checking source folders...")
    ok = True
    ok &= check_folder(IMAGE_REAL_FOLDER, "IMAGE_REAL_FOLDER")
    ok &= check_folder(IMAGE_FAKE_FOLDER, "IMAGE_FAKE_FOLDER")
    ok &= check_folder(VIDEO_REAL_FOLDER, "VIDEO_REAL_FOLDER")
    ok &= check_folder(VIDEO_FAKE_FOLDER, "VIDEO_FAKE_FOLDER")

    if not ok:
        print("\n❌ Fix the folder paths above then re-run this script.")
        return

    # Step 2: Clear output if it already exists (fresh merge)
    if OUTPUT_REAL.exists() or OUTPUT_FAKE.exists():
        print("\n[2] Clearing existing output folder (dataset/train/)...")
        shutil.rmtree("dataset", ignore_errors=True)
        print("  ✅ Cleared")
    else:
        print("\n[2] Output folder does not exist yet — will create fresh")

    # Step 3: Copy images
    print("\n[3] Copying image dataset...")
    real_imgs = copy_images(IMAGE_REAL_FOLDER, OUTPUT_REAL, "Real images")
    fake_imgs = copy_images(IMAGE_FAKE_FOLDER, OUTPUT_FAKE, "Fake images")

    # Step 4: Extract video frames
    print("\n[4] Extracting video frames...")
    real_frames = extract_video_frames(VIDEO_REAL_FOLDER, OUTPUT_REAL, "Real videos")
    fake_frames = extract_video_frames(VIDEO_FAKE_FOLDER, OUTPUT_FAKE, "Fake videos")

    # Step 5: Final summary
    total_real = real_imgs + real_frames
    total_fake = fake_imgs + fake_frames
    total      = total_real + total_fake

    print("\n" + "="*55)
    print("  DATASET READY ✅")
    print("="*55)
    print(f"  Real samples : {total_real:,}  ({real_imgs:,} images + {real_frames:,} video frames)")
    print(f"  Fake samples : {total_fake:,}  ({fake_imgs:,} images + {fake_frames:,} video frames)")
    print(f"  Total        : {total:,}")
    print(f"\n  Output saved to: dataset/train/")
    print(f"                   ├── real/  ({total_real:,} files)")
    print(f"                   └── fake/  ({total_fake:,} files)")
    print("\n  ✅ Now run:  python train.py --epochs 5")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()