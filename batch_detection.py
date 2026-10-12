"""
Run ball detection on every image in a folder, sorting results into
"detected" and "not_detected" subfolders, and printing a summary.

Usage:
    python test.py input_folder
    python test.py input_folder --conf 0.25 --batch_size 2
"""

import argparse
import os
import cv2
from ultralytics import YOLO
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Default path to your fine-tuned weights


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = str(BASE_DIR / "training_ses1_dataset" / "detect" / "10-12_Model_results" / "weights" / "best.pt")


def run_batch(input_dir, weights_path=DEFAULT_WEIGHTS, output_dir="results", conf=0.25, batch_size=2, imgsz=960):
    detected_dir = os.path.join(output_dir, "detected")
    not_detected_dir = os.path.join(output_dir, "not_detected")
    os.makedirs(detected_dir, exist_ok=True)
    os.makedirs(not_detected_dir, exist_ok=True)

    # Load your fine-tuned custom model
    print(f"Loading custom weights from: {weights_path}")
    model = YOLO(weights_path)

    image_files = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Running detection at imgsz={imgsz} in batches of {batch_size}...\n")

    detected_count = 0
    confidences = []

    for i in range(0, len(image_files), batch_size):
        batch_filenames = image_files[i:i + batch_size]
        batch_paths = [os.path.join(input_dir, f) for f in batch_filenames]

        results = model.predict(
            batch_paths,
            conf=conf,
            imgsz=imgsz,
            show=False,
            save=False,
            verbose=False,
        )

        for filename, r in zip(batch_filenames, results):
            has_ball = len(r.boxes) > 0
            out_dir = detected_dir if has_ball else not_detected_dir

            annotated = r.plot()
            cv2.imwrite(os.path.join(out_dir, filename), annotated)

            if has_ball:
                detected_count += 1
                box = r.boxes.cpu().numpy()[0]  # Highest-confidence detection
                x1, y1, x2, y2 = box.xyxy[0]
                box_conf = box.conf[0]
                confidences.append(box_conf)
                print(f"[DETECTED]     {filename}: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}), confidence={box_conf:.2f}")
            else:
                print(f"[NOT DETECTED] {filename}")

    print("\n--- Summary ---")
    print(f"Total images:     {len(image_files)}")
    print(f"Ball detected:    {detected_count}")
    print(f"No ball found:    {len(image_files) - detected_count}")
    if confidences:
        print(f"Average confidence (detected only): {sum(confidences) / len(confidences):.2f}")
    print(f"\nSorted images saved to: {output_dir}/detected and {output_dir}/not_detected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run ball detection on a folder of images.")
    parser.add_argument("input_dir", help="Folder containing images to process")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to best.pt weights")
    parser.add_argument("--output_dir", default="results", help="Folder to save sorted results into")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--batch_size", type=int, default=2, help="Images per batch (default: 2 for CPU)")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference resolution (default: 960)")
    args = parser.parse_args()

    run_batch(
        args.input_dir,
        args.weights,
        args.output_dir,
        args.conf,
        args.batch_size,
        args.imgsz,
    )