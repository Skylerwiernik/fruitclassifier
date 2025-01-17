import os
import cv2
from ultralytics import YOLO
from pathlib import Path


def crop_image(image_path:str|Path, output_dir:str|Path, model_weights:str|Path):
    """
    Creates detection from image
    :param image_path: path to the image
    :param output_dir: path to output directory
    :param model_weights: .pt filepath for model weights
    :return: list of filepaths to cropped images
    """
    model = YOLO(model_weights)
    image = cv2.imread(image_path)
    image = cv2.flip(image, 1)
    results = model.predict(image, iou=0.5)

    epsilon = 0
    crop_image_paths = list()
    index = 0
    for result in results:
        boxes = result.boxes
        scores = result.boxes.conf
        labels = result.boxes.cls
        for box, score, label in zip(boxes, scores, labels):
            # Get coordinates for bounding box
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            print(f"{label}_{score}")

            # Add padding to each of the coordinates
            x_min = max(x_min - epsilon, 0)
            y_min = max(y_min - epsilon, 0)
            x_max = min(x_max + epsilon, image.shape[1] - 1)
            y_max = min(y_max + epsilon, image.shape[0] - 1)

            # Extract the detected object
            detection = image[y_min:y_max, x_min:x_max]

            # Save the detection
            output_img_path = os.path.join(output_dir, f'{index}.jpg')
            cv2.imwrite(output_img_path, detection)
            cv2.imwrite(f'./detections/{index}.jpg', detection)
            crop_image_paths.append(output_img_path)
            index += 1

    return crop_image_paths


# # Example Usage
# image_path = './shots/fruit.jpg'
# output_directory = './detections/'
# model_weights = './models/YOLO_weights.pt'
# print(crop_image(image_path, output_directory, model_weights))