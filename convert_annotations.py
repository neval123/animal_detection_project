import os
import pandas as pd
from PIL import Image


def convert_oid_to_yolo(dataset_path):
    classes = ['Cat', 'Horse', 'Dog', 'Bird', 'Elephant', 'Fish', 'Hamster', 'Rabbit', 'Lion', 'Monkey']
    class_map = {cls.lower(): idx for idx, cls in enumerate(classes)}

    # Iteracja po klasach
    for class_name in classes:
        class_dir = os.path.join(dataset_path, 'validation', class_name)
        label_path = os.path.join(class_dir, 'Label')
        if not os.path.isdir(class_dir):
            continue
        if not os.path.isdir(label_path):
            continue
        # Iteracja po plikach z labelami
        for label_file in os.listdir(label_path):
            if label_file.endswith('.txt'):
                img_file = label_file.replace('.txt', '.jpg')
                img_path = os.path.join(class_dir, img_file)

                # Rozmiary obrazu
                with Image.open(img_path) as img:
                    img_width, img_height = img.size


                yolo_annotations = []
                with open(os.path.join(label_path, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_name = parts[0].lower()
                        xmin, ymin, xmax, ymax = map(float, parts[1:5])

                        # Konwersja do formatu YOLO
                        x_center = (xmin + xmax) / 2 / img_width
                        y_center = (ymin + ymax) / 2 / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height

                        # Numer klasy i koordynaty w formacie YOLO
                        yolo_line = f"{class_map[class_name]} {x_center} {y_center} {width} {height}"
                        yolo_annotations.append(yolo_line)

                # Zapis plików z labelami w formacie YOLO
                yolo_label_path = os.path.join(class_dir, 'labels')
                os.makedirs(yolo_label_path, exist_ok=True)
                with open(os.path.join(yolo_label_path, label_file), 'w') as f:
                    f.write('\n'.join(yolo_annotations))


# Ścieżka do datasetu
#dataset_path = './OIDv4_ToolKit/OID/Dataset'
dataset_path = './OID/Dataset'
convert_oid_to_yolo(dataset_path)
