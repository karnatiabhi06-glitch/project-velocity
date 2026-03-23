import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self):
        self.prev_gray = None
        self.transforms = [] # Store trajectory

    def stabilize(self, frame):
        """
        Calculate affine transform to stabilize frame against previous frame.
        Returns: Stabilized Frame, Transform Matrix
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return frame
            
        # Feature detection
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        
        if prev_pts is None:
             self.prev_gray = curr_gray
             return frame

        # Optical Flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None)
        
        # Filter valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if len(prev_pts) < 4:
             self.prev_gray = curr_gray
             return frame
             
        # Estimate Affine Transform (Translation + Rotation)
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        
        if m is None:
             self.prev_gray = curr_gray
             return frame
             
        # Extract translation (dx, dy) and angle (da)
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        
        # We want to INVERSE this movement to get a stable frame
        # Construct inverse transform matrix
        # Simplify: Just correct translation for now (Translation Stabilization) to avoid rotation artifacts
        # Full stabilization is complex, simple translation fix handles wind jitter well enough.
        
        # Create transformation matrix to shift image back by -dx, -dy
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        
        # Warp
        w, h = frame.shape[1], frame.shape[0]
        stabilized_frame = cv2.warpAffine(frame, M, (w, h))
        
        self.prev_gray = curr_gray
        
        return stabilized_frame
