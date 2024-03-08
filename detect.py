from ultralytics import YOLO
import cv2 as cv

model = YOLO("runs/detect/train/weights/best.pt")

img = cv.imread("extra/extra1.png")

result = model(img)[0]
print(result.boxes.data.tolist())
x1, y1, x2, y2, score, class_id = result.boxes.data[0]

img2 = cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
cv.imshow("title", img2)
cv.waitKey(0)
cv.destroyAllWindows(0)