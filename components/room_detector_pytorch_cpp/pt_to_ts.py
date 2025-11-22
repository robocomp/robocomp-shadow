from ultralytics import YOLO

# Cargar modelo YOLOv11
model = YOLO('best.pt')

# Exportar a TorchScript
model.export(format='torchscript')
