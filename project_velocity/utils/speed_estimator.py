import math
from collections import deque

class SpeedEstimator:
    def __init__(self, ppm=10, fps=30, buffer_size=5):
        self.ppm = ppm # Pixels Per Meter
        self.fps = fps
        self.previous_centroids = {}
        self.speed_buffer = {} # Stores deque for each objectID
        self.buffer_size = buffer_size
        
    def estimate_speed(self, objectID, centroid):
        speed = 0
        if objectID in self.previous_centroids:
            prev_centroid = self.previous_centroids[objectID]
            
            # Calculate pixel distance (Euclidean)
            d_pixels = math.sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2)
            
            # Conversion
            d_meters = d_pixels / self.ppm
            
            # Speed = Distance / (1/FPS) = Distance * FPS
            speed_mps = d_meters * self.fps
            
            # Convert to KMPH
            speed_kmph = speed_mps * 3.6
            
            # Smoothing Logic
            if objectID not in self.speed_buffer:
                self.speed_buffer[objectID] = deque(maxlen=self.buffer_size)
            
            self.speed_buffer[objectID].append(speed_kmph)
            
            # Calculate Average
            avg_speed = sum(self.speed_buffer[objectID]) / len(self.speed_buffer[objectID])
            
            speed = round(avg_speed, 2)
            
        self.previous_centroids[objectID] = centroid
        return speed

import cv2
import numpy as np

class HomographySpeedEstimator:
    def __init__(self, fps=30, buffer_size=8, src_points=None, dst_points=None):
        self.fps = fps
        self.buffer_size = buffer_size
        self.previous_locations = {} # using (x, y) meters
        self.speed_buffer = {}
        
        if src_points is not None and dst_points is not None:
             # Type Casting for OpenCV strictness
             src = np.array(src_points, dtype=np.float32)
             dst = np.array(dst_points, dtype=np.float32)
             self.H = cv2.getPerspectiveTransform(src, dst)
        else:
             print("Warning: Homography points not provided.")
             self.H = np.eye(3)

    def transform_point(self, point):
        # Point is (x, y)
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(p, self.H)
        return dst[0][0] # Returns (x_meters, y_meters)

    def estimate_speed(self, objectID, centroid):
        speed = 0
        current_meters = self.transform_point(centroid)
        
        if objectID in self.previous_locations:
            prev_meters = self.previous_locations[objectID]
            
            # Euclidean distance in meters
            d_meters = np.linalg.norm(current_meters - prev_meters)
            
            # Filter Jitter: If movement is tiny (< 5cm), ignore it to prevent noise at standstill
            if d_meters < 0.05:
                speed_kmph = 0
            else:
                # Speed = Distance * FPS (m/s)
                speed_mps = d_meters * self.fps
                speed_kmph = speed_mps * 3.6
            
            # --- 1. Max Speed Cap (Sanity Check) ---
            if speed_kmph > 200:
                # Outlier: Ignore massive spikes, return last known valid speed
                if objectID in self.speed_buffer and len(self.speed_buffer[objectID]) > 0:
                     avg = sum(self.speed_buffer[objectID]) / len(self.speed_buffer[objectID])
                     # Don't update buffer with bad value, just return old avg
                     self.previous_locations[objectID] = current_meters
                     return round(avg, 2)
                else: 
                     self.previous_locations[objectID] = current_meters
                     return 0

            # --- 2. Smoothing & Outlier Rejection ---
            if objectID not in self.speed_buffer:
                self.speed_buffer[objectID] = deque(maxlen=self.buffer_size)
            
            # Get current average (before adding new value) to check for sudden jumps
            current_avg = 0
            if len(self.speed_buffer[objectID]) > 0:
                current_avg = sum(self.speed_buffer[objectID]) / len(self.speed_buffer[objectID])
            
            # If we have history, check for massive acceleration (e.g. +60km/h in 1 frame)
            if len(self.speed_buffer[objectID]) >= 2:
                if abs(speed_kmph - current_avg) > 60:
                     # Treat as noise, assume constant velocity from last frame instead
                     speed_kmph = current_avg

            self.speed_buffer[objectID].append(speed_kmph)
            avg_speed = sum(self.speed_buffer[objectID]) / len(self.speed_buffer[objectID])
            
            speed = round(avg_speed, 2)
            
        self.previous_locations[objectID] = current_meters
        return speed
