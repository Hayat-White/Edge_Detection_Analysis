from ultralytics import YOLO
#Grab model
model = YOLO("yolov8n.pt")

results = model.train(data="data_set.yaml", epochs=10, imgsz=640)
