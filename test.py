from baseballcv.functions import LoadTools
from ultralytics import YOLO

# Downloads the pre-trained ball detection model
load_tools = LoadTools()
model_path = load_tools.load_model("ball_tracking")  # baseball-only detector

# Run inference
model = YOLO(model_path)
results = model.predict("image.png", show=True, save=True)

# Extract detections
for r in results:
    for box in r.boxes.cpu().numpy():
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        print(f"Ball found at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}), confidence={conf:.2f}")



