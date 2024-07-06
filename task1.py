import cv2
from ultralytics import YOLO


image_path = r'C:\Users\ftox\PycharmProjects\OPenCV\720x.png'
image = cv2.imread(image_path)

model_path = 'yolov5xu.pt'
model = YOLO(model_path)

results = model(image)

person_count = 0
for result in results:
    for box in result.boxes:
        if int(box.cls) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            person_count += 1


cv2.putText(image, f'Peoples: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.imwrite('output.jpg', image)

