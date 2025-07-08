import time
import numpy as np
import open3d as o3d
from src.model import BodyModel


class BodyModelVisualizer:
    def __init__(self, model: BodyModel, update_interval: float=0):
        self.model = model
        self.update_interval = update_interval

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="LBLM - VIS")

        # Set black background
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.0, 0.0, 0.0])

        self.joint_spheres = []
        self.joint_markers = []
        self.line_set = o3d.geometry.LineSet()
        self._init_geometry()
        self._running = True

    def _init_geometry(self):

        # init a bounding box to enclose the model
        box = o3d.geometry.TriangleMesh.create_box(width=100,height=100, depth=100)
        lines = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        lines.colors = o3d.utility.Vector3dVector([[0.5, 0, 0] for _ in range(len(lines.lines))])
        self.vis.add_geometry(lines)

        # get the first 3 coordinates of each row
        coords = np.array(self.model.data)[:, :3]
        colors = self.model.data[:, 3]

        for pos, color in zip(coords, colors):
            # Main red joint sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(pos)
            sphere.paint_uniform_color([color, color, color])
            self.joint_spheres.append(sphere)
            self.vis.add_geometry(sphere)

        self.joint_connections = [
            # head
            (0, 1),  # nose to left eye (inner)
            (1, 2),  # left eye (inner) to left eye
            (2, 3),  # left eye to left eye (outer)
            (0, 4),  # nose to right eye (inner)
            (4, 5),  # right eye (inner) to right eye
            (5, 6),  # right eye to right eye (outer)
            (3, 7),  # left eye (outer) to left ear
            (6, 8),  # right eye (outer) to right ear
            (9, 10),  # mouth (left) to mouth (right)

            # torso and arms
            (11, 12),  # left shoulder to right shoulder
            (11, 13),  # left shoulder to left elbow
            (13, 15),  # left elbow to left wrist
            (15, 17),  # left wrist to left pinky
            (15, 19),  # left wrist to left index
            (15, 21),  # left wrist to left thumb

            (12, 14),  # right shoulder to right elbow
            (14, 16),  # right elbow to right wrist
            (16, 18),  # right wrist to right pinky
            (16, 20),  # right wrist to right index
            (16, 22),  # right wrist to right thumb

            # hips and legs
            (23, 24),  # left hip to right hip

            (23, 25),  # left hip to left knee
            (25, 27),  # left knee to left ankle
            (27, 29),  # left ankle to left heel
            (27, 31),  # left ankle to left foot index
            (29, 31),  # left heel to left foot index

            (24, 26),  # right hip to right knee
            (26, 28),  # right knee to right ankle
            (28, 30),  # right ankle to right heel
            (28, 32),  # right ankle to right foot index
            (30, 32)  # right heel to right foot index
        ]

        lines = o3d.utility.Vector2iVector(self.joint_connections)
        points = o3d.utility.Vector3dVector(coords)
        self.line_set.points = points
        self.line_set.lines = lines
        self.line_set.colors = o3d.utility.Vector3dVector([[1,1,1] for _ in self.joint_connections])
        self.vis.add_geometry(self.line_set)

    def update(self):
        coords = self.model.data[:, :3]
        colors = self.model.data[:, 3]

        for sphere, pos, color in zip(self.joint_spheres, coords, colors):
            pass
            #offset = pos - sphere.get_center()
            #sphere.translate(offset, relative=False)
            #sphere.paint_uniform_color([color, color, color])
            #self.vis.update_geometry(sphere)

        self.line_set.points = o3d.utility.Vector3dVector(coords)
        self.vis.update_geometry(self.line_set)
        time.sleep(self.update_interval)

    def run(self):
        while self._running:
            self.update()
            self.vis.poll_events()
            self.vis.update_renderer()

        self.vis.destroy_window()

    def stop(self):
        self._running = False
