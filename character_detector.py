import cv2
import os
import sys
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

class CharacterDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def detect_characters(self, image_path):
        img = Image.open(image_path).convert('RGB')
        results = self.model(img, size=640)
        return results.pandas().xyxy[0]

    def display_image_with_boxes(self, image_path, detections, save_path=None):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]

        detection_list = []

        if save_path:
            label_file = save_path.replace('.jpg', '.txt')
            with open(label_file, 'w') as file:
                for index, row in detections.iterrows():
                    x_center, y_center, width, height = self.convert_bbox_to_yolo_format(
                        row['xmin'], row['ymin'], row['xmax'], row['ymax'], img_width, img_height
                    )
                    detection_list.append([0, x_center, y_center, width, height]) 
                    file.write(f"{int(row['class'])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        for index, row in detections.iterrows():
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Image and label saved to {save_path} and {label_file}")
        for detection in detection_list:
            print(detection)
    @staticmethod
    def convert_bbox_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return x_center, y_center, width, height

def main():
    if len(sys.argv) != 4:
        print("Usage: python character_detector.py model_path image_path save_path")
        sys.exit(1)

    model_path, image_path, save_path = sys.argv[1], sys.argv[2], sys.argv[3]
    detector = CharacterDetector(model_path)
    detections = detector.detect_characters(image_path)

    if not detections.empty:
        detector.display_image_with_boxes(image_path, detections, save_path)
    else:
        print("No characters detected.")

if __name__ == "__main__":
    main()

