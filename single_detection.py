from baseballcv.functions import LoadTools
from ultralytics import YOLO
from pathlib import Path

# Downloads the pre-trained ball detection model
load_tools = LoadTools()
model_path = load_tools.load_model("ball_tracking") 


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = str(BASE_DIR / "training_ses1_dataset" / "detect" / "10-12_Model_results" / "weights" / "best.pt")


# Run inference
model = YOLO(DEFAULT_WEIGHTS)
results = model.predict("MicrosoftTeams-video (1).mp4", show=True, save=True)

# Extract detections
for r in results:
    for box in r.boxes.cpu().numpy():
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        print(f"Ball found at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}), confidence={conf:.2f}")


