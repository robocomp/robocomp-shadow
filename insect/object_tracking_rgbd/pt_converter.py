from ultralytics import YOLO

# Load a model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO('yolov8n-seg.pt')  # load an official model

# Export the model
model.export(format='engine', device='0')
