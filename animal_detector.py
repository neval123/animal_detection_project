import torch
import cv2



class AnimalDetector:
    def __init__(self, model_path=None): #'./runs/train/exp3/weights/best.pt'

        # Wczytanie modelu
        if model_path:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

        # Ustawienie parametrów detekcji
        self.model.conf = 0.25  # Współczynnik pewności
        self.model.iou = 0.45  # Współczynnik NMS IOU

        # Klasy zwierząt
        self.animal_classes = [
            'bird', 'cat', 'dog', 'horse', 'elephant',
            'fish', 'lion', 'monkey', 'hamster', 'rabbit'
        ]

    def detect_animals(self,
                       image_path, save_results=True):

        # Detekcja
        results = self.model(image_path)

        detections = []
        for detection in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
            x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
            class_name = results.names[int(cls)]

            # Tylko wybrane klasy
            if class_name.lower() in self.animal_classes:
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })

        if save_results:
            results.save()  # 'runs/detect/exp{n}'

        return detections

    def draw_detections(self, image_path, detections, output_path):

        image = cv2.imread(image_path)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf = det['confidence']

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(output_path, image)
