"""
    Convert GLB animations to searchable vectors using Panda3D and MediaPipe.
"""


import math
import os
import shutil

import numpy as np
from PIL import Image

from lblm.detection.live_detector import Detector

WIDTH = 720
HEIGHT = 1280

from panda3d.core import loadPrcFileData

loadPrcFileData("", f"win-size {WIDTH} {HEIGHT} ")  # Set the window size
loadPrcFileData("", "aspect-ratio 0.5625000246")  # set the aspect ratio to 9:16

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4, Filename
from direct.actor.Actor import Actor


class Converter(ShowBase):
    def __init__(self, options):
        # Disable Panda3D's automatic clock advancement
        loadPrcFileData("", "clock-mode limited")  # or "clock-mode none" to disable entirely
        loadPrcFileData("", "sync-video false")  # disable vsync to avoid framerate throttling

        ShowBase.__init__(self)

        # init the animations and the base animation
        self.recorded_index = 0
        self.frame_index = 0
        self.frame_rate = 30

        self.disableMouse()
        self.set_background_color(0, 0, 0, 1)

        # Load GLB model with rig
        self.animations = options

        self.actor = Actor("vis/model.glb", self.animations)
        self.actor.reparentTo(self.render)
        self.actor.setScale(1)
        self.actor.setPos(0, 0, 0)
        self.actor.stop()

        # Lights
        alight = AmbientLight("ambient")
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        dlight = DirectionalLight("dlight")
        dlight.setColor(Vec4(0.8, 0.8, 0.8, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

        # Camera (frontal, centered)
        bounds = self.actor.getTightBounds()
        center = (bounds[0] + bounds[1]) * 0.5
        cam_dist = 3  # How far in front of the model

        self.camera.setPos(center.getX(), center.getY() - cam_dist, center.getZ())
        self.camera.lookAt(center)

        self.warmup_frames = 5  # Number of frames to warm up the animation

        self.current_vec: np.ndarray | None = None

        self.detector = Detector(None, None)
        self.detector.initialize_mediapipe()

        # Start animation cycling
        self.taskMgr.add(self.record, "RecordTask")



    def record(self, task):
        if self.warmup_frames > 0:
            self.warmup_frames -= 1
            return task.cont

        anim_name = list(self.animations.keys())[self.recorded_index]
        if os.path.exists(f"animations/vectors/{anim_name}.npy"):
            print(f"Animation {anim_name} already recorded. Skipping.")
            self.recorded_index += 1
            return task.cont

        anim_name = list(self.animations.keys())[self.recorded_index]
        animation_limit = math.floor(self.actor.getDuration(anim_name) * self.frame_rate)

        if self.frame_index >= animation_limit:
            print(f"Animation {anim_name} completed. Total frames: {self.frame_index + 1}")
            self.current_vec *= 1 / (self.frame_index + 1)
            np.save(f"animations/vectors/{anim_name}.npy", self.current_vec)

            # remove the frames after video creation
            shutil.rmtree(f"animations/videos/{anim_name}/frames")

            self.recorded_index += 1
            self.frame_index = 0
            self.warmup_frames = 5

            anim_name = list(self.animations.keys())[self.recorded_index]
            print(f"Recording animation: {anim_name}")
            return task.cont

        # advance the animation
        self.actor.pose(anim_name, frame=self.frame_index)

        os.makedirs(f"animations/videos/{anim_name}/frames", exist_ok=True)  # Create directory if it doesn't exist

        filename = Filename(f"animations/videos/{anim_name}/frames/{self.frame_index}.png")
        self.win.saveScreenshot(filename)
        self.frame_index += 1

        img = Image.open(str(filename)).convert("RGB")
        _,model = self.detector.process_frame(np.array(img))
        if self.current_vec is None:
            self.current_vec = model.landmarks
        else:
            self.current_vec += model.landmarks

        return task.cont
