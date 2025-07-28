"""
    Convert GLB animations to MP4 videos using Panda3D and MoviePy.
    This script initializes a Panda3D application, loads a GLB model with animations,
    and records each animation as a video file in MP4 format.
    Can be executed from the command line.
"""


import math
import os
import shutil

from PIL import Image
from moviepy import ImageSequenceClip
from lblm.animations import load_animations

WIDTH = 720
HEIGHT = 1280

from panda3d.core import loadPrcFileData

loadPrcFileData("", f"win-size {WIDTH} {HEIGHT} ")  # Set the window size
loadPrcFileData("", "aspect-ratio 0.5625000246")  # set the aspect ratio to 9:16

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4, Filename
from direct.actor.Actor import Actor


class Converter(ShowBase):
    def __init__(self, options:dict[str, str], model_path="data/character.glb"):
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

        self.actor = Actor(model_path, self.animations)
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

        # Start animation cycling
        self.taskMgr.add(self.record, "RecordTask")

    def record(self, task):
        if self.warmup_frames > 0:
            self.warmup_frames -= 1
            return task.cont

        anim_name = list(self.animations.keys())[self.recorded_index]
        if os.path.exists(f"animations/videos/{anim_name}.mp4"):
            print(f"Animation {anim_name} already recorded. Skipping.")
            self.recorded_index += 1
            return task.cont

        anim_name = list(self.animations.keys())[self.recorded_index]
        animation_limit = math.floor(self.actor.getDuration(anim_name) * self.frame_rate)

        if self.frame_index >= animation_limit:
            print(f"Animation {anim_name} completed. Total frames: {self.frame_index + 1}")
            self.images_to_video_moviepy(f"animations/videos/{anim_name}/frames", f"animations/videos/{anim_name}.mp4")
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

        img = Image.open(str(filename))
        img = img.resize((WIDTH, HEIGHT))  # match your window size
        img.save(str(filename))

        return task.cont

    def images_to_video_moviepy(self, image_dir, output):
        try:
            files = [
                f
                for f in os.listdir(image_dir)
                if f.endswith(".png")
            ]
            files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
            files = [os.path.join(image_dir, f) for f in files]
            clip = ImageSequenceClip(files, fps=self.frame_rate)
            clip.write_videofile(output, codec="libx264")
        except Exception as e:
            print(f"Error creating video: {e}")

if __name__ == "__main__":
    actor_path = input("Path to actor model (GLB Model Required): ")
    animations_path = input("Path to animation models (GLB Models Required): ")

    animations = load_animations(animations_path=animations_path)

    converter = Converter(options=animations, model_path=actor_path)
    converter.run()
