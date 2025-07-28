import ssl
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue, Event

import cv2
import mediapipe as mp
import numpy as np


"""Fix SSL context for MediaPipe model downloads"""
try:
    # Try to create unverified HTTPS context
    ssl._create_default_https_context = ssl._create_unverified_context
    print("SSL context fixed for MediaPipe downloads")
except Exception as e:
    print(f"Could not fix SSL context: {e}")


@dataclass
class BodyModel:
    """Dataclass to store body pose landmarks with certainty values"""
    landmarks: np.ndarray = None  # 4D array: [landmark_id, x, y, z, visibility]
    frame_width: int = 0
    frame_height: int = 0
    timestamp: float = 0.0
    process_time: float = 0.0

    def __post_init__(self):
        if self.landmarks is None:
            # Initialize with 33 pose landmarks, each with [x, y, z, visibility]
            self.landmarks = np.zeros((33, 4), dtype=np.float32)


class Detector(Process):
    """Complete body landmark detection system running in separate process"""

    def __init__(self, data_queue: Queue, stop_event: Event):
        super().__init__()
        self.data_queue = data_queue
        self.stop_event = stop_event

        # Will be initialized in the process
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose = None

        # Pose landmark names for reference
        self.landmark_names = [
            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
            "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
            "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
            "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
        ]

        self.show_angles = False
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

        self.quantization_chunk_length = 0
        self.quantization_start = 0
        self.chunk_amount = 0
        self.chunk: BodyModel | None = None
        self.reset_chunks()

    def reset_chunks(self):
        self.quantization_chunk_length = 6
        self.quantization_start = time.time()
        self.chunk_amount = 0
        self.chunk = None

    def initialize_mediapipe(self):
        """Initialize MediaPipe in the process"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """Process a single frame and return landmarks data"""
        start_time = time.time()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(rgb_frame)

        # Create body data
        body_data = BodyModel()
        body_data.frame_width = frame.shape[1]
        body_data.frame_height = frame.shape[0]
        body_data.timestamp = time.time()
        body_data.process_time = time.time() - start_time

        # Extract landmarks
        if results.pose_landmarks:
            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Update dataclass with landmark data
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                body_data.landmarks[idx] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ]
        return frame, body_data

    def get_body_info(self, body_data: BodyModel):
        """Get current body landmarks information as string"""
        info_lines = []
        info_lines.append(f"Frame: {body_data.frame_width}x{body_data.frame_height}")
        info_lines.append(f"Process time: {body_data.process_time * 1000:.1f}ms")
        info_lines.append(f"FPS: {self.fps:.1f}")

        # Check if any landmarks are detected
        if np.any(body_data.landmarks):
            info_lines.append("Body detected!")

            # Show key landmarks with high visibility
            key_landmarks = [
                (0, "NOSE"), (11, "L_SHOULDER"), (12, "R_SHOULDER"),
                (13, "L_ELBOW"), (14, "R_ELBOW"), (15, "L_WRIST"), (16, "R_WRIST"),
                (23, "L_HIP"), (24, "R_HIP"), (25, "L_KNEE"), (26, "R_KNEE")
            ]

            visible_count = 0
            for idx, name in key_landmarks:
                x, y, z, visibility = body_data.landmarks[idx]
                if visibility > 0.5:
                    info_lines.append(f"{name}: ({x:.3f}, {y:.3f}, {z:.3f}) vis:{visibility:.3f}")
                    visible_count += 1
                    if visible_count >= 4:  # Limit display to avoid clutter
                        break
        else:
            info_lines.append("No body detected")

        return info_lines

    def get_body_angles(self, body_data: BodyModel):
        """Calculate some basic body angles"""
        if not np.any(body_data.landmarks):
            return []

        angles = []

        # Left arm angle
        l_shoulder = body_data.landmarks[11][:2]
        l_elbow = body_data.landmarks[13][:2]
        l_wrist = body_data.landmarks[15][:2]

        if all(body_data.landmarks[i][3] > 0.5 for i in [11, 13, 15]):
            v1 = l_shoulder - l_elbow
            v2 = l_wrist - l_elbow
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            angles.append(f"L_ARM_ANGLE: {np.degrees(angle):.1f}°")

        # Right arm angle
        r_shoulder = body_data.landmarks[12][:2]
        r_elbow = body_data.landmarks[14][:2]
        r_wrist = body_data.landmarks[16][:2]

        if all(body_data.landmarks[i][3] > 0.5 for i in [12, 14, 16]):
            v1 = r_shoulder - r_elbow
            v2 = r_wrist - r_elbow
            angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
            angles.append(f"R_ARM_ANGLE: {np.degrees(angle):.1f}°")

        return angles

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time

    def run(self):
        """Main detection loop running in separate process"""
        print("Starting body landmark detection process...")

        # Initialize MediaPipe in this process
        self.initialize_mediapipe()

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Body Landmark Detection Started in separate process.")
        print("Controls: 'q' to quit, 'p' to print landmark data, 'a' to toggle angles")

        while not self.stop_event.is_set():
            ret, frame = cap.read()

            # rotate the frame by 90deg
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process frame
            processed_frame, body_data = self.process_frame(frame)

            # Update FPS
            self.update_fps()

            # Get info and angle data
            info_lines = self.get_body_info(body_data)
            if self.show_angles:
                angle_lines = self.get_body_angles(body_data)
                info_lines.extend(angle_lines)

            # Add info text overlay
            y_offset = 30
            for line in info_lines:
                cv2.putText(processed_frame, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

            # Add control instructions
            cv2.putText(processed_frame, "Press 'q':quit, 'p':print data, 'a':toggle angles",
                        (10, processed_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Send data to main process (non-blocking)
            try:
                tracked_time = time.time() - self.quantization_start
                if tracked_time > self.quantization_chunk_length:
                    if self.chunk is not None:
                        self.chunk.landmarks *= 1 / self.chunk_amount  # Average landmarks
                        self.data_queue.put(self.chunk, block=False)
                        print("Quantized detection chunk sent to main process")
                        self.reset_chunks()
                else:
                    if self.chunk is None:
                        self.chunk = body_data
                        self.quantization_start = time.time()
                    else:
                        self.chunk.landmarks += body_data.landmarks
                        self.chunk_amount += 1
            except Exception as e:

                pass  # Queue full, skip

            # Display frame
            cv2.imshow('Body Landmark Detection', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop_event.set()
                break
            elif key == ord('p'):
                # Print detailed landmark data
                print("\n=== Current Body Landmark Data ===")
                print(f"Shape: {body_data.landmarks.shape}")
                print("Landmarks with visibility > 0.5:")
                for i, (x, y, z, vis) in enumerate(body_data.landmarks):
                    if vis > 0.5:
                        name = self.landmark_names[i] if i < len(self.landmark_names) else f"LANDMARK_{i}"
                        print(f"{i:2d} {name:15s}: ({x:.3f}, {y:.3f}, {z:.3f}) vis:{vis:.3f}")
                print("=" * 50)
            elif key == ord('a'):
                self.show_angles = not self.show_angles
                print(f"Angle display: {'ON' if self.show_angles else 'OFF'}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.pose:
            self.pose.close()
        print("Body landmark detection process stopped")


class BodyLandmarkSystem:
    """Main system that manages the detection process"""

    def __init__(self):
        self.data_queue = Queue(maxsize=10)
        self.stop_event = Event()
        self.detection_process = None
        self.latest_data = None

    def start_detection(self):
        """Start the detection process"""
        if self.detection_process is not None and self.detection_process.is_alive():
            print("Detection process is already running")
            return

        self.stop_event.clear()
        detector = Detector(self.data_queue, self.stop_event)
        self.detection_process = Process(target=detector.run)
        self.detection_process.start()
        print("Detection process started")

    def stop_detection(self):
        """Stop the detection process"""
        if self.detection_process is None:
            print("No detection process running")
            return

        self.stop_event.set()
        self.detection_process.join(timeout=5)

        if self.detection_process.is_alive():
            print("Force terminating detection process")
            self.detection_process.terminate()
            self.detection_process.join()

        print("Detection process stopped")

    def get_latest_data(self):
        """Get the latest landmark data (non-blocking)"""
        try:
            while not self.data_queue.empty():
                self.latest_data = self.data_queue.get_nowait()
        except:
            pass
        return self.latest_data

    def is_running(self):
        """Check if detection process is running"""
        return self.detection_process is not None and self.detection_process.is_alive()

    def run_interactive(self):
        """Run interactive mode in main process"""
        print("Body Landmark Detection System")
        print("Commands:")
        print("  'start' - Start detection")
        print("  'stop' - Stop detection")
        print("  'status' - Check status")
        print("  'data' - Print latest landmark data")
        print("  'quit' - Exit system")

        while True:
            try:
                command = input("\nEnter command: ").strip().lower()

                if command == 'start':
                    self.start_detection()
                elif command == 'stop':
                    self.stop_detection()
                elif command == 'status':
                    if self.is_running():
                        print("Detection process is running")
                    else:
                        print("Detection process is not running")
                elif command == 'data':
                    data = self.get_latest_data()
                    if data:
                        print(f"Latest data - Frame: {data.frame_width}x{data.frame_height}, "
                              f"Process time: {data.process_time * 1000:.1f}ms")
                        detected_landmarks = np.count_nonzero(data.landmarks[:, 3] > 0.5)
                        print(f"Detected landmarks: {detected_landmarks}/33")
                    else:
                        print("No data available")
                elif command == 'quit':
                    break
                else:
                    print("Unknown command")

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except EOFError:
                print("\nEOF received")
                break

        # Cleanup
        self.stop_detection()


def main():
    """Main function - runs detection directly or interactive mode"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Interactive mode - control from main process
        system = BodyLandmarkSystem()
        system.run_interactive()
    else:
        # Direct mode - run detection in separate process
        data_queue = Queue(maxsize=10)
        stop_event = Event()

        detector = Detector(data_queue, stop_event)
        detector.start()

        try:
            # Main process can do other work here
            # For demo, just wait and occasionally check data
            while detector.is_alive():
                time.sleep(1)

                # Example: Get latest data in main process
                try:
                    while not data_queue.empty():
                        latest_data = data_queue.get_nowait()
                        print(f"Main process received data: {latest_data.timestamp:.3f} - "
                              f"Landmarks: {np.count_nonzero(latest_data.landmarks[:, 3] > 0.5)}/33")
                        # Process data in main process if needed
                        # print(f"Main process received data: {latest_data.timestamp}")
                except:
                    pass

        except KeyboardInterrupt:
            print("\nMain process interrupted")
            stop_event.set()

        detector.join()
        print("Main process finished")
