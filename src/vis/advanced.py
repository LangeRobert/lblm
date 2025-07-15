from panda3d.core import loadPrcFileData, Shader

from src.vis.pie import PieChart3D

loadPrcFileData('', 'basic-shaders-only #f')  # Allow custom shaders

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4
from direct.actor.Actor import Actor
from direct.task import Task
from direct.interval.LerpInterval import LerpFunc
from multiprocessing import Process, Queue
import sys


class Visualizer(Process):

    def __init__(self, options, queue: Queue):
        super().__init__()
        self.options = options
        self.queue = queue

    def run(self):
        try:
            vis = _Visualizer(self.options, self.queue)
            vis.start()
        except Exception as e:
            print(f"Error in visualizer process: {e}")
            import traceback
            traceback.print_exc()


class _Visualizer(ShowBase):
    def __init__(self, options, queue: Queue):
        try:
            ShowBase.__init__(self)
            self.queue = queue

            # init the animations and the base animation
            self.current_anim_index = 0
            print(f"Available animations: {list(options.keys())}")

            self.disableMouse()
            self.set_background_color(0, 0, 0, 1)

            # Load GLB model with rig
            self.animations = options

            self.actor = Actor("vis/model.glb", self.animations)
            self.actor.reparentTo(self.render)
            self.actor.setScale(1)
            self.actor.setPos(0, 0, 0)

            # pie chart
            self.pie = PieChart3D(self.render)
            self.pie_duration = 8.0  # seconds for full animation
            self.taskMgr.add(self.update_pie, "update_pie")

            # Load and apply the shader
            try:
                shader = Shader.load(Shader.SL_GLSL, "vis/shader/gradient.vert", "vis/shader/gradient.frag")
                if shader:
                    self.actor.setShader(shader)
                else:
                    print("Warning: Could not load shader")
            except Exception as e:
                print(f"Warning: Shader loading failed: {e}")

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
            if bounds:
                center = (bounds[0] + bounds[1]) * 0.5
                cam_dist = 2  # How far in front of the model
                self.camera.setPos(center.getX(), center.getY() - cam_dist, center.getZ())
                self.camera.lookAt(center)
            else:
                print("Warning: Could not get actor bounds, using default camera position")
                self.camera.setPos(0, -5, 0)
                self.camera.lookAt(0, 0, 0)

        except Exception as e:
            print(f"Error initializing visualizer: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def animate(self, task: Task):
        try:
            from_anim = list(self.animations.keys())[self.current_anim_index]

            if self.queue.empty():
                to_anim_index = 0
            else:
                value = self.queue.get(block=False)
                if value and value in self.animations:
                    to_anim_index = list(self.animations.keys()).index(value)
                else:
                    to_anim_index = 0

            to_anim = list(self.animations.keys())[to_anim_index]
            self.current_anim_index = to_anim_index

            self.actor.enableBlend()  # Enable animation blending
            self.actor.loop(from_anim)
            self.actor.loop(to_anim)

            self.actor.setControlEffect(from_anim, 1.0)
            self.actor.setControlEffect(to_anim, 0.0)

            # Create a crossfade by interpolating weights
            def set_blend(t):
                self.actor.setControlEffect(from_anim, 1.0 - t)
                self.actor.setControlEffect(to_anim, t)

            # Crossfade over 1 second
            LerpFunc(set_blend, fromData=0.0, toData=1.0, duration=1.0).start()

            # get the length of the current animation
            if to_anim_index == 0:
                anim_length = 2
            else:
                anim_length = min(self.actor.getDuration(to_anim), 8)

            # Schedule next transition after animation length
            self.taskMgr.doMethodLater(anim_length, self.animate, "AnimateTask")
            return Task.done
        except Exception as e:
            print(f"Error in animate task: {e}")
            import traceback
            traceback.print_exc()
            return Task.done

    def start(self):
        try:
            # Schedule animation switching immediately
            self.taskMgr.doMethodLater(0, self.animate, "AnimateTask")
            # Start the main loop
            self.run()
        except Exception as e:
            print(f"Error starting visualizer: {e}")
            import traceback
            traceback.print_exc()

    def update_pie(self, task):
        elapsed = task.time
        self.pie.update(elapsed % self.pie_duration / self.pie_duration)
        return Task.cont


# Alternative approach - run in main thread instead of subprocess
class VisualizerMainThread:
    def __init__(self, options, queue: Queue):
        self.options = options
        self.queue = queue
        self.vis = None

    def start(self):
        try:
            self.vis = _Visualizer(self.options, self.queue)
            self.vis.start()
        except Exception as e:
            print(f"Error in visualizer: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        if self.vis:
            self.vis.userExit()