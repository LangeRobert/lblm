from queue import Queue

from panda3d.core import loadPrcFileData, Shader

loadPrcFileData('', 'basic-shaders-only #f')  # Allow custom shaders

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4
from direct.actor.Actor import Actor
from direct.task import Task
from direct.interval.LerpInterval import LerpFunc

class Visualizer(ShowBase):
    def __init__(self, options, queue:Queue):
        ShowBase.__init__(self)
        self.queue = queue
        self.current_anim_index = 0

        self.disableMouse()
        self.set_background_color(0, 0, 0, 1)

        # Load GLB model with rig
        self.animations = options


        self.actor = Actor("vis/models/model.glb",self.animations)
        self.actor.reparentTo(self.render)
        self.actor.setScale(1)
        self.actor.setPos(0, 0, 0)

        # Load and apply the shader
        shader = Shader.load(Shader.SL_GLSL, "vis/shader/gradient.vert", "vis/shader/gradient.frag")
        self.actor.setShader(shader)

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
        self.taskMgr.doMethodLater(0, self.cycle_animations, "CycleAnimationsTask")

    def cycle_animations(self, task: Task):
        from_anim = list(self.animations.keys())[self.current_anim_index]

        if self.queue.empty():
            to_anim_index = 0
        else:
            value = self.queue.get(block=False)
            if value:
                print(f"🎭 Received animation request: {value}")
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
        anim_length = self.actor.getDuration(to_anim)

        # Schedule next transition after 5 seconds
        self.taskMgr.doMethodLater(anim_length, self.cycle_animations, "CycleAnimationsTask")
        return Task.done

    def start(self):
        # Schedule animation switching every 5 seconds
        self.taskMgr.doMethodLater(5, self.cycle_animations, "CycleAnimationsTask")
