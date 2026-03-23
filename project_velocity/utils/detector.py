import cv2
import numpy as np

class VehicleDetector:
    def __init__(self, weights_path, config_path, classes_path, threshold=0.5, nms_threshold=0.2):
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        
        # Load classes
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
            
        # Vehicle classes we are interested in (COCO indices or names)
        # COCO names usually include: 'car', 'motorcycle', 'bus', 'truck'
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
        
        # Load Model
        self.net = cv2.dnn_DetectionModel(weights_path, config_path)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
        
    def detect(self, frame):
        classIds, confs, bbox = self.net.detect(frame, confThreshold=self.threshold)
        
        results = []
        if len(classIds) != 0:
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))
            indices = cv2.dnn.NMSBoxes(bbox, confs, self.threshold, self.nms_threshold)
            
            for i in indices:
                # Handle different OpenCV versions for NMSBoxes return
                idx = i
                if isinstance(i, (tuple, list, np.ndarray)):
                    idx = i[0]
                idx = int(idx)
                
                # Robustly get classID
                # classIds might be a list or numpy array [N, 1] or [N]
                class_id = classIds[idx]
                if isinstance(class_id, (list, np.ndarray)):
                     class_id = class_id[0]
                class_id = int(class_id)
                
                # Check bounds for Class Names
                # COCO classes are usually 1-91, or 0-80 depending on file
                # self.class_names is 0-indexed list
                # Usually MobileNet returns 1-based index (1 = person)
                # So name is class_names[class_id - 1]
                
                if 0 <= class_id - 1 < len(self.class_names):
                    class_name = self.class_names[class_id - 1]
                else:
                    # Fallback or Skip
                    # print(f"Warning: ClassID {class_id} out of range (0-{len(self.class_names)})")
                    continue
                
                # Filter only vehicles
                if class_name in self.vehicle_classes:
                    box = bbox[idx]
                    confidence = confs[idx]
                    results.append((box, class_name, confidence))
                    
        return results
