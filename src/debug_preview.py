"""Optional main-thread OpenCV input preview, owned by the command-line session."""

from src.m_0_camera.contract import CameraFrame
from src.m_1_pose_estimator.contract import PoseEstimate

POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


class DebugPreview:
    """Display captured input without opening another camera or retaining frames."""

    def __init__(self) -> None:
        """Create a lazy preview; no GUI resources are acquired yet.

        :returns: None.
        """
        self._name = "LBLM input — debug"
        self._opened = False
        self._disabled = False

    def show(self, frame: CameraFrame, estimate: PoseEstimate) -> None:
        """Display the captured pixels and pump HighGUI events on the main thread.

        :param frame: Packed RGB, BGR or RGBA input in its captured orientation.
        :param estimate: Image-space landmarks inferred from this exact frame.
        :returns: None; Escape or closing the window disables only the preview.
        """
        if estimate.source.frame_id != frame.frame_id:
            raise ValueError("Overlay and image must refer to the same frame")
        if self._disabled:
            return
        import cv2
        import numpy as np

        channels = 4 if frame.pixel_format == "rgba8" else 3
        pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(
            frame.height, frame.width, channels
        )
        if frame.pixel_format == "rgb8":
            pixels = np.asarray(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), dtype=np.uint8)
        elif frame.pixel_format == "rgba8":
            pixels = np.asarray(cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR), dtype=np.uint8)
        pixels = pixels.copy()  # Drawing must not mutate the immutable camera payload.
        points: dict[int, tuple[int, int]] = {}
        for i, joint in enumerate(estimate.image_joints):
            p = joint.position
            if p is not None and 0 <= p.x <= 1 and 0 <= p.y <= 1:
                points[i] = (round(p.x * (frame.width - 1)), round(p.y * (frame.height - 1)))
        if estimate.skeleton.skeleton_id == "mediapipe-world-33-v1":
            for a, b in POSE_CONNECTIONS:
                if a in points and b in points:
                    strong = (
                        min(
                            estimate.image_joints[a].confidence, estimate.image_joints[b].confidence
                        )
                        >= 0.3
                    )
                    cv2.line(
                        pixels,
                        points[a],
                        points[b],
                        (70, 220, 70) if strong else (100, 100, 100),
                        2,
                    )
        for i, point in points.items():
            color = (70, 255, 70) if estimate.image_joints[i].confidence >= 0.3 else (0, 165, 255)
            cv2.circle(pixels, point, 4, color, -1)
        status = (
            "No person detected"
            if estimate.pose is None
            else "Detected joints: green=visible, orange=uncertain"
        )
        cv2.putText(
            pixels, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            pixels, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        if not self._opened:
            cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)
            self._opened = True
        elif cv2.getWindowProperty(self._name, cv2.WND_PROP_VISIBLE) < 1:
            self.close()
            return
        cv2.imshow(self._name, pixels)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close()

    def close(self) -> None:
        """Destroy only this session's preview, including on failure or cancellation.

        :returns: None; safe to repeat.
        """
        self._disabled = True
        if self._opened:
            import cv2

            self._opened = False
            try:
                cv2.destroyWindow(self._name)
                cv2.waitKey(1)
            except cv2.error:
                # A user may already have closed the native window.
                pass
