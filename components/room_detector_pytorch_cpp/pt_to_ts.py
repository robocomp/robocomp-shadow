from ultralytics import YOLO

# Cargar modelo YOLOv11
model = YOLO('best.pt')

# Exportar a Torchscript
model.export(
    format='onnx',
    dynamic=False,
    simplify=True,
    opset=12
)
