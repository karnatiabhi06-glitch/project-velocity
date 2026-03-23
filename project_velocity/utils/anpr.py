try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("EasyOCR not installed. ANPR disabled.")
    EASYOCR_AVAILABLE = False

import cv2
import numpy as np
import os
from utils.detector_plate import PlateDetector

class LicensePlateReader:
    def __init__(self):
        if not EASYOCR_AVAILABLE:
            self.reader = None
            return
            
        # Initialize Reader for English. Use GPU if available.
        # This might be slow on first load.
        print("Initializing EasyOCR... (This may take a moment)")
        self.reader = easyocr.Reader(['en'], gpu=True) 
        
        # Initialize Plate Detector
        print("Initializing Plate Detector...")
        try:
            self.plate_detector = PlateDetector(model_path='plate_best.pt', conf=0.20)
        except Exception as e:
            print(f"Warning: Could not load PlateDetector ({e}). Fallback to full crop (less accurate).")
            self.plate_detector = None

    def read_plate(self, frame, vehicle_bbox):
        """
        Crop the vehicle/bbox and attempt to read text.
        Returns: Best text string or None
        """
        if self.reader is None:
            return None
            
        x, y, w, h = vehicle_bbox
        
        # Ensure bounds
        h_img, w_img = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w < 20 or h < 10:
            return None
            
        # 1. Crop Vehicle
        vehicle_crop = frame[y:y+h, x:x+w]
        
        ocr_region = vehicle_crop
        
        # 2. Detect Plate inside Vehicle (if detector available)
        if self.plate_detector:
            plate_box = self.plate_detector.detect_plate(vehicle_crop)
            if plate_box:
                px, py, pw, ph = plate_box
                # Add some padding to plate crop if needed
                pad = 2
                px = max(0, px - pad)
                py = max(0, py - pad)
                pw = min(pw + 2*pad, w - px)
                ph = min(ph + 2*pad, h - py)
                
                ocr_region = vehicle_crop[py:py+ph, px:px+pw]
                # Debug: cv2.imshow("Plate", ocr_region); cv2.waitKey(1)
        
        # 3. OCR
        try:
            # gray = cv2.cvtColor(ocr_region, cv2.COLOR_BGR2GRAY) # EasyOCR does this internally
            results = self.reader.readtext(ocr_region)
            
            # Filter results
            detected_text = []
            for (bbox, text, prob) in results:
                if prob > 0.3: 
                    # Basic filter: Alphanumeric and length > 2
                    clean_text = ''.join(e for e in text if e.isalnum())
                    if len(clean_text) > 2:
                        detected_text.append(clean_text.upper())
            
            if detected_text:
                return "".join(detected_text) # Return joined text
            
        except Exception as e:
            print(f"OCR Error: {e}")
            
        return None
