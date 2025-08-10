# app/vision.py

import cv2
import numpy as np
from ultralytics import YOLO

class VisionSystem:
    def __init__(self, person_model='yolov8n.pt', face_model='yolov8n-face.pt', skip_frames=75):
        # Load models
        self.person_model = YOLO(person_model)
        self.face_model = YOLO(face_model)
        self.mode = 'full_body'
        self.target_id = None
        self.model = self.person_model
        
        # Frame skipping
        self._frame_counter = 0
        self._skip_frames = skip_frames
        self._last_results = None
        
        # Last known positions
        self._last_person_box = None
        self._last_face_box = None
        
        # Debug flag
        self.debug_draw_face = True

    def _resize_frame(self, frame, target_width=720, target_height=480):
        # Resize frame and calculate scales
        orig_h, orig_w = frame.shape[:2]
        resized = cv2.resize(frame, (target_width, target_height))
        scale_x = orig_w / target_width
        scale_y = orig_h / target_height
        return resized, scale_x, scale_y

    def set_mode(self, mode: str):
        # Set tracking mode
        self.mode = mode if mode in ['face', 'torso', 'full_body'] else 'full_body'

    def select_person(self, frame, click_x, click_y):
        # Select person by click
        small_frame, scale_x, scale_y = self._resize_frame(frame)
        results = self.person_model.track(small_frame, classes=[0], persist=True, verbose=False)[0]
        
        if results.boxes is None or results.boxes.id is None or results.boxes.xyxy is None:
            return
        
        boxes = (results.boxes.xyxy.cpu().numpy() * np.array([scale_x, scale_y, scale_x, scale_y])).astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int)
        
        for bbox, tid in zip(boxes, ids):
            x1, y1, x2, y2 = bbox
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.target_id = int(tid)
                break

    def _detect_face_in_person(self, frame, person_box):
        x1, y1, x2, y2 = person_box
        person_crop = frame[y1:y2, x1:x2]
        
        # Debug: Check crop size
        crop_h, crop_w = person_crop.shape[:2]
        print(f"[FACE DEBUG] Person crop size: {crop_w}x{crop_h}")
        
        face_results = self.face_model.predict(person_crop, verbose=False, conf=0.3)[0]
        
        if face_results.boxes is not None and len(face_results.boxes) > 0:
            face_box = face_results.boxes.xyxy[0].cpu().numpy().astype(int)
        else:
            return None


    def _estimate_torso_from_face(self, face_box, person_box):
        # Estimate torso from face and person
        face_x1, face_y1, face_x2, face_y2 = face_box
        person_x1, person_y1, person_x2, person_y2 = person_box
        
        face_height = face_y2 - face_y1
        torso_height = int(face_height * 3)
        torso_width = int((person_x2 - person_x1) * 0.8)
        
        face_center_x = (face_x1 + face_x2) // 2
        torso_x1 = max(person_x1, face_center_x - torso_width // 2)
        torso_x2 = min(person_x2, face_center_x + torso_width // 2)
        torso_y1 = face_y1
        torso_y2 = min(person_y2, face_y1 + torso_height)
        
        return np.array([torso_x1, torso_y1, torso_x2, torso_y2])

    def process_frame(self, frame):
        # Process frame for detection
        small_frame, scale_x, scale_y = self._resize_frame(frame)
        
        if (self._frame_counter % (self._skip_frames + 1) == 0) or self._last_results is None:
            results = self.person_model.track(small_frame, classes=[0], persist=True, verbose=False)[0]
            self._last_results = results
        else:
            results = self._last_results
        
        self._frame_counter += 1
        
        if results.boxes is None or results.boxes.id is None or results.boxes.xyxy is None:
            if self._last_person_box is not None:
                return frame.copy(), self._last_person_box, []
            return frame.copy(), None, []
        
        boxes = (results.boxes.xyxy.cpu().numpy() * np.array([scale_x, scale_y, scale_x, scale_y])).astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int)
        
        person_box = None
        tracking_box = None
        
        if self.target_id is not None:
            for bbox, tid in zip(boxes, ids):
                if tid == self.target_id:
                    person_box = bbox
                    break
        
        if person_box is None and self._last_person_box is not None:
            person_box = self._last_person_box
        
        if person_box is not None:
            self._last_person_box = person_box
            
            if self.mode == 'full_body':
                tracking_box = person_box
                
            elif self.mode == 'face':
                face_box = self._detect_face_in_person(frame, person_box)
                if face_box is not None:
                    tracking_box = face_box
                    self._last_face_box = face_box
                    self._person_box_for_zoom = person_box
                elif self._last_face_box is not None:
                    tracking_box = self._last_face_box
                    self._person_box_for_zoom = person_box
                else:
                    h = person_box[3] - person_box[1]
                    face_height = int(h * 0.25)
                    tracking_box = np.array([person_box[0], person_box[1], person_box[2], person_box[1] + face_height])
                    self._person_box_for_zoom = person_box
                    
            elif self.mode == 'torso':
                face_box = self._detect_face_in_person(frame, person_box)
                if face_box is not None:
                    tracking_box = self._estimate_torso_from_face(face_box, person_box)
                else:
                    h = person_box[3] - person_box[1]
                    torso_height = int(h * 0.6)
                    tracking_box = np.array([person_box[0], person_box[1], person_box[2], person_box[1] + torso_height])
        
        # Draw overlays
        annotated = frame.copy()
        
        for bbox, tid in zip(boxes, ids):
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if tid == self.target_id else (180, 180, 180)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.putText(annotated, f"ID{tid}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if tracking_box is not None:
            x1, y1, x2, y2 = tracking_box.astype(int)
            mode_colors = {'full_body': (0, 0, 255), 'face': (0, 255, 255), 'torso': (0, 165, 255)}
            color = mode_colors.get(self.mode, (0, 0, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, self.mode.upper(), (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated, tracking_box, ids.tolist()
