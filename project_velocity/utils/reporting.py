import csv
import cv2
import os
import datetime

class Reporter:
    def __init__(self, log_file='traffic_log.csv', snapshot_dir='overspeeding/cars'):
        self.log_file = log_file
        self.snapshot_dir = snapshot_dir
        
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
            
        # Initialize CSV with header if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'VehicleID', 'VehicleType', 'Speed_KMPH', 'Overspeed'])
        
        # Cache to prevent spamming logs for same vehicle
        self.logged_vehicles = {} # {id: last_log_time}
        self.log_interval = 2.0 # Seconds between logs for same vehicle

    def log_vehicle(self, vehicle_id, speed, is_overspeed, frame=None, bbox=None, vehicle_type="Vehicle"):
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if recently logged
        if vehicle_id in self.logged_vehicles:
            last_time = self.logged_vehicles[vehicle_id]
            if (now - last_time).total_seconds() < self.log_interval:
                return # Skip logging to reduce lag
        
        self.logged_vehicles[vehicle_id] = now
        
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, vehicle_id, vehicle_type, speed, 'YES' if is_overspeed else 'NO'])
        except Exception as e:
            print(f"[Warning] Failed to write log: {e}")
            
        if is_overspeed and frame is not None and bbox is not None:
             # Snapshot logic remains same, but maybe limited?
             pass # Keeping snapshot logic below
        
        if is_overspeed and frame is not None and bbox is not None:
            try:
                x, y, w, h = bbox
                # Ensure crop is within bounds
                h_img, w_img = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                
                crop = frame[y:y+h, x:x+w]
                if crop.size > 0:
                    filename = f"{self.snapshot_dir}/ID_{vehicle_id}_{timestamp.replace(':','-')}.jpg"
                    cv2.imwrite(filename, crop)
            except Exception as e:
                print(f"[Warning] Failed to save snapshot: {e}")
