from ultralytics import YOLO
import json

# Load a pretrained YOLOv8 model (automatically downloads)
# 3x larger than "model = YOLO('yolov8n.pt')" but much more accurate
model = YOLO('yolov8s.pt')  # Small (slower, more accurate)

# Run detection on an image
results = model.predict('test_image1.jpeg')

# Extract detected objects
for result in results:
    for box in result.boxes:
        object_name = result.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"Found: {object_name} (confidence: {confidence:.2f})")