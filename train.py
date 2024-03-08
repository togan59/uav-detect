import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8s.yaml")
results = model.train(data = "data.yaml", epochs = 20)
results = model.val()
model.export(format="onnx")