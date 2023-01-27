from ultralytics import YOLO


model = YOLO("yolov8s.yaml")  
model = YOLO("yolov8s.pt")

# Train the model
results = model.train(data="data.yaml", epochs=100, imgsz=640, plots=True, save=True)