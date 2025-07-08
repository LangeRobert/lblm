import threading
import time

import cv2
import numpy as np
from mediapipe.tasks import python
import mediapipe as mp

from src.model import BodyModel

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "./detection/model.task"

class Detector(threading.Thread):

    def __init__(self, model:BodyModel):
        super().__init__()
        self.model = model
        self.daemon = True


    def callback(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        landmarks = result.pose_world_landmarks
        if landmarks:
            state = landmarks.pop()

            """
            JOINT INDEXES:
            0 - nose
            1 - left eye (inner)
            2 - left eye
            3 - left eye (outer)
            4 - right eye (inner)
            5 - right eye
            6 - right eye (outer)
            7 - left ear
            8 - right ear
            9 - mouth (left)
            10 - mouth (right)
            11 - left shoulder
            12 - right shoulder
            13 - left elbow
            14 - right elbow
            15 - left wrist
            16 - right wrist
            17 - left pinky
            18 - right pinky
            19 - left index
            20 - right index
            21 - left thumb
            22 - right thumb
            23 - left hip
            24 - right hip
            25 - left knee
            26 - right knee
            27 - left ankle
            28 - right ankle
            29 - left heel
            30 - right heel
            31 - left foot index
            32 - right foot index
            """
            # Convert to numpy array for easier manipulation
            values = np.array([[val.x, -val.y, -val.z, val.visibility] for val in state])
            self.model.data = values * 100  # Scale to millimeters

    def run(self):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.callback)

        with PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                landmarker.detect_async(mp_image, int(time.time_ns() / 1_000_000))  # Convert to milliseconds
                if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                    break

            cap.release()


if __name__ == '__main__':
    model = BodyModel.default()
    detector = Detector(model)
    detector.start()
    detector.join()  # Wait for the thread to finish




