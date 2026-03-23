from ultralytics import YOLO
import cv2
import numpy as np
import os

class PlateDetector:
    def __init__(self, model_path='plate_best.pt', conf=0.25):
        """
        Initialize YOLOv8 Plate Detector.
        model_path: Path to .pt file (relative to this file's parent or absolute)
        """
        # Resolve model path relative to project root or current dir
        if not os.path.exists(model_path):
             # Try looking one level up if in utils
             model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
             
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_plate(self, vehicle_image):
        """
        Detect license plate in a cropped vehicle image.
        Returns: proper bounding box (x, y, w, h) relative to the vehicle_image, or None
        """
        if vehicle_image is None or vehicle_image.size == 0:
            return None
            
        results = self.model(vehicle_image, verbose=False, conf=self.conf)
        
        best_box = None
        max_conf = -1
        
        for result in results:
            for box in result.boxes:
                # We assume the model only has one class or we take the one with highest confidence
                c = float(box.conf[0])
                if c > max_conf:
                    max_conf = c
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    best_box = (x1, y1, w, h)
                    
        return best_box
