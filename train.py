"""
Fine-tune BaseballCV's pretrained ball_tracking model on your own labeled photos.

Requires a YOLO-format dataset (e.g. Roboflow export) with a data.yaml.

Usage:
    python train.py data.yaml
    python train.py data.yaml --imgsz 960 --batch 4 --epochs 100
    python train.py data.yaml --batch -1          # let Ultralytics pick batch size for your GPU
"""

import argparse

import torch
from baseballcv.functions import LoadTools
from ultralytics import YOLO


def fine_tune(data_yaml, epochs, imgsz, batch, freeze, scale, patience, name):
    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found. Training at this image size on CPU will be very slow.\n")

    # Start from the pretrained ball_tracking weights (fine-tuning, not from scratch).
    model_path = LoadTools().load_model("ball_tracking")
    model = YOLO(model_path)

    print(f"Starting fine-tune from: {model_path}")
    print(f"Dataset: {data_yaml} | epochs={epochs} imgsz={imgsz} batch={batch} "
          f"freeze={freeze} scale={scale}\n")

# Recommended YOLO training command
    model.train(
    data="data.yaml",
    epochs=100,
    imgsz=960,
    lr0=0.001,  # Lower learning rate to prevent weight destruction
    freeze=10,  # Freeze backbone features
    mosaic=0.0,  # Prevent small objects from being cropped out
    device='cpu',  # Force GPU (use 'cpu' only if no CUDA available)
    batch=8,
)

    save_dir = model.trainer.save_dir
    print("\nTraining complete.")
    print(f"Best weights: {save_dir}/weights/best.pt")
    print("Test them with:  python eval.py --data data.yaml --split valid "
          f"--weights {save_dir}/weights/best.pt --imgsz {imgsz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune the ball_tracking model on custom data.")
    ap.add_argument("data_yaml", help="Path to your dataset's data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960, help="Training image size (default 960)")
    ap.add_argument("--batch", type=int, default=4, help="Batch size; use -1 for auto")
    ap.add_argument("--freeze", type=int, default=10, help="Backbone layers to freeze (0 = none)")
    ap.add_argument("--scale", type=float, default=0.2, help="Scale augmentation (default 0.2)")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--name", default="ball_tracking_finetuned")
    args = ap.parse_args()

    fine_tune(args.data_yaml, args.epochs, args.imgsz, args.batch,
              args.freeze, args.scale, args.patience, args.name)