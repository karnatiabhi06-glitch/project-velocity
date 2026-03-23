import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # Color Palette (BGR)
        self.COLOR_NORMAL = (255, 255, 0)   # Cyan
        self.COLOR_DANGER = (0, 0, 255)     # Red
        self.COLOR_TEXT = (255, 255, 255)   # White
        self.COLOR_BG = (0, 0, 0)           # Black
        self.COLOR_ROI = (0, 255, 0)        # Matrix Green for ROI
        self.COLOR_ROI_FILL = (0, 50, 0)    # Dark Green for ROI Fill

    def draw_corner_rect(self, frame, bbox, color, thickness=2, length=20):
        x, y, w, h = bbox
        
        # Draw a faint full box for context
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        
        # Draw thick corners (Bracket Style)
        # Top Left
        cv2.line(frame, (x, y), (x + length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + length), color, thickness)
        
        # Top Right
        cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
        
        # Bottom Left
        cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
        
        # Bottom Right
        cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)
        
        return frame

    def draw_info_card(self, frame, bbox, text, color):
        x, y, w, h = bbox
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        font_thickness = 2
        
        (fw, fh), baseline = cv2.getTextSize(text, font, scale, font_thickness)
        
        # Card Position (Bottom of box for a change, or Top) - Let's keep Top but sleek
        rect_x = x
        rect_y = y - fh - 15
        if rect_y < 0: rect_y = y + 10
        
        rect_w = fw + 20
        rect_h = fh + 20
        
        # High-Tech Background (Black with color border)
        # 1. Background
        sub_img = frame[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            white_rect = np.full_like(sub_img, self.COLOR_BG)
            res = cv2.addWeighted(sub_img, 0.3, white_rect, 0.7, 1.0)
            frame[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = res

        # 2. Border around label
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color, 1)
        
        # 3. Text
        cv2.putText(frame, text, (rect_x + 10, rect_y + fh + 8), font, scale, self.COLOR_TEXT, 1) # Text shadow/glow simulation?
        cv2.putText(frame, text, (rect_x + 10, rect_y + fh + 8), font, scale, color, 1) 
        
        return frame

    def draw_roi(self, frame, points):
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        # 1. Region Overlay (Transparent Fill)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], self.COLOR_ROI_FILL)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # 2. Polygon Outline (Dotted effect simulated by Polylines for now, or just Solid Tech Line)
        cv2.polylines(frame, [pts], True, self.COLOR_ROI, 2)
        
        # 3. Vertices (Circles)
        for pt in pts:
            cv2.circle(frame, tuple(pt[0]), 5, self.COLOR_ROI, -1)
            cv2.circle(frame, tuple(pt[0]), 8, self.COLOR_ROI, 1)
            
        # 4. Label "SCANNING ZONE" at top-left-most point
        top_left = min(points, key=lambda p: p[0] + p[1])
        cv2.putText(frame, " // SCANNING ZONE", (top_left[0] + 10, top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_ROI, 1)
        
        return frame
