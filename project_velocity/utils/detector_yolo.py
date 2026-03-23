from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', classes=[2, 3, 5, 7]):
        """
        Initialize YOLOv8 Detector.
        model_path: Path to .pt file (will download if not present)
        classes: List of COCO class IDs to detect (2=car, 3=motorcycle, 5=bus, 7=truck)
        """
        self.model = YOLO(model_path)
        self.classes = classes

    def detect(self, frame, conf_threshold=0.4):
        """
        Detect vehicles in the frame.
        Returns list of (box, class_name, confidence)
        """
        # Optimization: Use imgsz=256 for faster inference if frame is small
        imgsz = 256 if frame.shape[1] <= 480 else 320
        try:
            results = self.model(frame, verbose=False, conf=conf_threshold, classes=self.classes, imgsz=imgsz)
        except Exception:
            # In cloud/webcam environments inference can occasionally fail on a frame.
            return []
        
        detections = []
        for result in results:
            for box in result.boxes:
                # Bounding Box (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                
                # Confidence
                conf = float(box.conf[0])
                
                # Class Name
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                # Format: (x, y, w, h) for consistency with old detector
                bbox = (x1, y1, w, h)
                
                detections.append((bbox, cls_name, conf))
                
        return detections
