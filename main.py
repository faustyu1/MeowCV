"""
Online Cat Mediapipe Program - MeowCV
Enhanced Version
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import sys
import time
from pathlib import Path

# --- CONFIGURATION ---
ASSETS_DIR = Path("assets")
PHOTOS_DIR = Path("photos")
MODEL_PATH = "face_landmarker.task"

# Thresholds
THRESHOLDS = {
    "TONGUE_STRONG": 0.45,
    "TONGUE_WEAK": 0.12,
    "MOUTH_DIST_STRONG": 0.09,
    "MOUTH_DIST_WEAK": 0.035,
    "SMILE": 0.4,
    "SHOCK": 0.35,
    "SQUINT": 0.5
}

SMOOTHING = 0.15

# --- ASSET MAPPING ---
# Key: Emotion name, Value: Filename
CAT_IMAGES = {
    "default": "larry.jpeg",
    "tongue": "cat-tongue.jpeg",
    "disgust": "cat-disgust.jpeg",
    "happy": "cat-happy.png",
    "shock": "cat-shock.jpeg",
    "glare": "cat-glare.jpeg"
}

class MeowCV:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.scores = {"j": 0.0, "m": 0.0, "s": 0.0, "q": 0.0, "h": 0.0}
        self.detector = self._init_detector()
        self.cam = self._init_camera()
        
        # Ensure directories exist
        if not PHOTOS_DIR.exists():
            PHOTOS_DIR.mkdir()

    def _init_detector(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _init_camera(self):
        print(f"Searching for camera starting from index {self.camera_index}...")
        for i in range(self.camera_index, self.camera_index + 5):
            for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, None]:
                cap = cv2.VideoCapture(i + (backend if backend else 0))
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"Success! Camera {i} opened.")
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        return cap
                    cap.release()
        print("FATAL: No working camera found.")
        sys.exit(1)

    def get_score(self, blendshapes, name):
        for category in blendshapes:
            if category.category_name == name:
                return category.score
        return 0.0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        current_cat = CAT_IMAGES["default"]
        debug_msg = "No Face Detected"

        if result.face_landmarks and result.face_blendshapes:
            landmarks = result.face_landmarks[0]
            blendshapes = result.face_blendshapes[0]

            # 1. Extract raw values
            face_h = abs(landmarks[10].y - landmarks[152].y)
            raw = {
                "j": self.get_score(blendshapes, 'jawOpen'),
                "m": abs(landmarks[13].y - landmarks[14].y) / face_h,
                "s": max(self.get_score(blendshapes, 'eyeWideLeft'), self.get_score(blendshapes, 'eyeWideRight')),
                "q": max(self.get_score(blendshapes, 'eyeSquintLeft'), self.get_score(blendshapes, 'eyeSquintRight')),
                "h": (self.get_score(blendshapes, 'mouthSmileLeft') + self.get_score(blendshapes, 'mouthSmileRight')) / 2.0
            }

            # 2. Smooth scores
            for k in self.scores:
                self.scores[k] = self.scores[k] * (1 - SMOOTHING) + raw[k] * SMOOTHING

            debug_msg = f"Jaw:{self.scores['j']:.1f} Mouth:{self.scores['m']:.2f} Happy:{self.scores['h']:.1f}"

            # 3. Decision Logic (Priority based)
            if self.scores['j'] > THRESHOLDS["TONGUE_STRONG"] or self.scores['m'] > THRESHOLDS["MOUTH_DIST_STRONG"]:
                current_cat = CAT_IMAGES["tongue"]
            elif self.scores['j'] > THRESHOLDS["TONGUE_WEAK"] or self.scores['m'] > THRESHOLDS["MOUTH_DIST_WEAK"]:
                current_cat = CAT_IMAGES["disgust"]
            elif self.scores['h'] > THRESHOLDS["SMILE"]:
                current_cat = CAT_IMAGES["happy"]
            elif self.scores['s'] > THRESHOLDS["SHOCK"]:
                current_cat = CAT_IMAGES["shock"]
            elif self.scores['q'] > THRESHOLDS["SQUINT"]:
                current_cat = CAT_IMAGES["glare"]

            # Draw landmarks
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)

        return frame, current_cat, debug_msg

    def run(self):
        cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Cat Image', cv2.WINDOW_NORMAL)
        
        print("\nMeowCV is running!")
        print("- Press SPACE to take a photo")
        print("- Press ESC to exit\n")

        while True:
            ret, frame = self.cam.read()
            if not ret: break

            processed_frame, cat_file, debug_msg = self.process_frame(frame)
            
            # Show debug info
            cv2.putText(processed_frame, debug_msg, (10, 30), 1, 1.2, (0, 255, 255), 2)
            cv2.imshow('Face Detection', processed_frame)

            # Load and show cat
            cat_path = ASSETS_DIR / cat_file
            cat_img = cv2.imread(str(cat_path))
            
            if cat_img is not None:
                try:
                    win = cv2.getWindowImageRect('Cat Image')
                    tw, th = max(win[2], 100), max(win[3], 100)
                except:
                    tw, th = 640, 480
                cat_resized = cv2.resize(cat_img, (tw, th))
                cv2.imshow("Cat Image", cat_resized)
            else:
                cat_resized = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(cat_resized, f"Missing: {cat_file}", (20, 50), 1, 1, (0,0,255), 1)
                cv2.imshow("Cat Image", cat_resized)

            # Controls
            key = cv2.waitKey(1)
            if key == 27: # ESC
                break
            elif key == 32: # SPACE
                self.take_photo(processed_frame, cat_img if cat_img is not None else cat_resized)

        self.cam.release()
        cv2.destroyAllWindows()

    def take_photo(self, frame, cat_img):
        h, w = frame.shape[:2]
        cat_side = cv2.resize(cat_img, (w, h))
        combined = np.hstack((frame, cat_side))
        
        filename = PHOTOS_DIR / f"meowcv_{int(time.time())}.png"
        cv2.imwrite(str(filename), combined)
        print(f"📸 Meow! Photo saved: {filename}")

if __name__ == "__main__":
    start_index = 0
    if len(sys.argv) > 1:
        try: start_index = int(sys.argv[1])
        except: pass
        
    app = MeowCV(camera_index=start_index)
    app.run()
