from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    def __init__(self, max_age=30):
        # deep_sort_realtime defaults to a PyTorch embedder ("mobilenet"), which
        # can fail on some cloud environments due to optional packaging deps
        # (e.g. pkg_resources/setuptools). Fall back to embedder=None to keep
        # tracking working without crashing WebRTC startup.
        try:
            self.tracker = DeepSort(max_age=max_age)
        except ModuleNotFoundError as e:
            if e.name == "pkg_resources":
                self.tracker = DeepSort(max_age=max_age, embedder=None)
            else:
                raise

    def update(self, detections, frame):
        """
        Update tracker with detections.
        detections: List of (bbox, class_name, confidence)
        bbox: (x, y, w, h)
        """
        # Format for DeepSort: [([left, top, w, h], conf, class_name), ...]
        formatted_dets = []
        for det in detections:
            bbox, cls_name, conf = det
            # bbox: (x, y, w, h) which is [left, top, w, h]
            formatted_dets.append((bbox, conf, cls_name))
            
        tracks = self.tracker.update_tracks(formatted_dets, frame=frame)
        
        objects = {} # Format: {ID: (centroid_x, centroid_y)} for compatibility
        # We also want bbox for visualization, so we might need to return more info
        # But for drop-in replacement, we return objects dict first.
        # Actually, let's keep it compatible with app.py Loop
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            # Fix for "Flying Particles": 
            # If the object wasn't detected in the current frame (just predicted), 
            # don't show it. This prevents ghosting when objects leave the screen.
            if track.time_since_update > 1:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom
            
            x1, y1, x2, y2 = map(int, ltrb)
            # Centroid
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            
            # Get class name if available
            # deep-sort-realtime stores the class in track.det_class
            cls_name = getattr(track, 'det_class', "Vehicle")
            if cls_name is None:
                cls_name = "Vehicle"
            
            # Get bbox in (x, y, w, h) format
            w = x2 - x1
            h = y2 - y1
            bbox = (x1, y1, w, h)
            
            # Format: {ID: (centroid, class_name, bbox)}
            objects[track_id] = ((cX, cY), cls_name, bbox)
            
        return objects
