from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s-seg.pt')  # load an official model

# Export the model
model.export(format='engine', device='0')
