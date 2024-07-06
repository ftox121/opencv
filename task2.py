import cv2
import numpy as np
from ultralytics import YOLO
from typing import Any, Optional, Tuple, List

def _euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """Вычисляет евклидово расстояние между двумя точками."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def _find_count_of_person(image: Any, boxes: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    """Определяет количество людей и их позиции на изображении."""
    num_people = 0
    people_centers = []
    people_boxes = []

    for i in range(len(boxes)):
        if int(class_ids[i]) == 0:  # class_id == 0 для людей
            num_people += 1
            x1, y1, x2, y2 = map(int, boxes[i])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            people_centers.append((center_x, center_y))
            people_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f'Person: {confidences[i]:.2f}'
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return num_people, people_centers, people_boxes

def find_groups(image: Any, people_centers: List[Tuple[int, int]], people_boxes: List[Tuple[int, int, int, int]], distance_threshold: int = 100) -> List[List[int]]:
    """Находит группы людей и рисует рамки вокруг групп."""
    groups = []
    visited = set()

    for i, center in enumerate(people_centers):
        if i not in visited:
            group = [i]
            visited.add(i)
            for j, other_center in enumerate(people_centers):
                if j != i and _euclidean_distance(center, other_center) < distance_threshold:
                    group.append(j)
                    visited.add(j)
            groups.append(group)

    for group in groups:
        if len(group) > 1:
            x1 = min(people_boxes[i][0] for i in group)
            y1 = min(people_boxes[i][1] for i in group)
            x2 = max(people_boxes[i][2] for i in group)
            y2 = max(people_boxes[i][3] for i in group)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f'Group of {len(group)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return groups

def find_person_on_image(image_path: str, result_path: str, model_path: str) -> None:
    """Определяет людей на изображении и находит группы."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    model = YOLO(model_path)
    results = model(image_path, conf=0.2)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    output_image = image.copy()

    num_people, people_centers, people_boxes = _find_count_of_person(output_image, boxes, confidences, class_ids)

    cv2.putText(output_image, f'Number of people: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    find_groups(output_image, people_centers, people_boxes)

    save_success = cv2.imwrite(result_path, output_image)
    if save_success:
        print(f"Image successfully saved to {result_path}")
    else:
        print(f"Failed to save image to {result_path}")

    cv2.imshow('Image with People and Groups Detected', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r'C:\Users\ftox\PycharmProjects\OPenCV\images\image.png'
    result_path = r'C:\Users\ftox\PycharmProjects\OPenCV\images\result1.png'
    model_path = 'yolov8s.pt'

    find_person_on_image(image_path, result_path, model_path)
