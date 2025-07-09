from panda3d.core import loadPrcFileData, Shader

loadPrcFileData('', 'basic-shaders-only #f')  # Allow custom shaders

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4, ClockObject
from direct.actor.Actor import Actor
from direct.task import Task
from direct.interval.LerpInterval import LerpFunc

class Visualizer(ShowBase):
    def __init__(self, options):
        ShowBase.__init__(self)
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
        to_anim_index = (self.current_anim_index + 1) % len(self.animations)
        to_anim = list(self.animations.keys())[to_anim_index]
        self.current_anim_index = to_anim_index

        print(f"🎞️ Crossfading from {from_anim} to {to_anim}")

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

        # Schedule next transition after 5 seconds
        self.taskMgr.doMethodLater(5.0, self.cycle_animations, "CycleAnimationsTask")
        return Task.done

    def start(self):
        # Schedule animation switching every 5 seconds
        self.taskMgr.doMethodLater(5, self.cycle_animations, "CycleAnimationsTask")

if __name__ == '__main__':
    app = Visualizer()
    app.run()
