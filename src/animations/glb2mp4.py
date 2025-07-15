import math
import os
from moviepy import ImageSequenceClip
import os

from panda3d.core import loadPrcFileData
loadPrcFileData("", "win-size 720 1280 ")       # Set the window size
loadPrcFileData("", "aspect-ratio 0.5625000246") # set the aspect ratio to 9:16


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

        self.actor = Actor("vis/model.glb",self.animations)
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
        cam_dist = 2  # How far in front of the model

        self.camera.setPos(center.getX(), center.getY() - cam_dist, center.getZ())
        self.camera.lookAt(center)

        # Start animation cycling
        self.taskMgr.add(self.record, "RecordTask")

    def record(self, task):
        anim_name = list(self.animations.keys())[self.recorded_index]
        animation_limit = math.floor(self.actor.getDuration(anim_name)*self.frame_rate)

        if self.frame_index == animation_limit-1:
            print(f"Animation {anim_name} completed. Total frames: {self.frame_index + 1}")
            self.images_to_video_moviepy(f"animations/videos/{anim_name}/frames",f"animations/videos/{anim_name}.mp4")
            # remove the frames after video creation
            os.rmdir(f"animations/videos/{anim_name}")


        if self.frame_index >= animation_limit:
            self.recorded_index += 1
            self.frame_index = 0
            anim_name = list(self.animations.keys())[self.recorded_index]
            print(f"Recording animation: {anim_name}")

        # advance the animation
        self.actor.pose(anim_name, frame=self.frame_index)

        os.makedirs(f"animations/videos/{anim_name}/frames", exist_ok=True)  # Create directory if it doesn't exist

        filename = Filename(f"animations/videos/{anim_name}/frames/{self.frame_index}.png")
        self.win.saveScreenshot(filename)
        self.frame_index += 1

        return task.cont


    @staticmethod
    def images_to_video_moviepy(image_dir, output, fps=30):
        files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ][1:])
        clip = ImageSequenceClip(files, fps=fps)
        clip.write_videofile(output, codec="libx264")
