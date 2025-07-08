from panda3d.core import loadPrcFileData, Shader

loadPrcFileData('', 'basic-shaders-only #f')  # Allow custom shaders

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4, ClockObject
from direct.actor.Actor import Actor
from direct.task import Task


class App(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.animations = ["idle", "punch"]
        self.current_anim_index = 0

        self.disableMouse()
        self.set_background_color(0, 0, 0, 1)

        # Load GLB model with rig
        self.actor = Actor("models/model_v1.glb",{anim: f"models/model_v1.glb" for anim in self.animations})  # Must be rigged+exported with animations
        self.actor.reparentTo(self.render)
        self.actor.setScale(1)
        self.actor.setPos(0, 0, 0)

        # Preload animations for blending
        for anim in self.animations:
            self.actor.loop(anim)
            self.actor.stop(anim)
            self.actor.setControlEffect(anim, 0.0)

        # Load and apply the shader
        shader = Shader.load(Shader.SL_GLSL, "shader/gradient.vert", "shader/gradient.frag")
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

    def crossfade_to_animation(self, new_anim, fade_duration=0.5):
        old_anim = self.animations[self.current_anim_index]
        self.current_anim_index = (self.current_anim_index + 1) % len(self.animations)

        self.actor.enableBlend()
        self.actor.setControlEffect(old_anim, 1.0)
        self.actor.setControlEffect(new_anim, 0.0)

        self.actor.loop(old_anim)
        self.actor.loop(new_anim)

        def blend_task(task):
            t = min(task.time / fade_duration, 1.0)
            self.actor.setControlEffect(old_anim, 1.0 - t)
            self.actor.setControlEffect(new_anim, t)
            if t >= 1.0:
                # Stop blending once transition is complete
                self.actor.stop(old_anim)
                self.actor.setControlEffect(old_anim, 0.0)
                self.actor.setControlEffect(new_anim, 1.0)
                return Task.done
            return Task.cont

        self.taskMgr.add(blend_task, "BlendTask")

    def cycle_animations(self, task: Task):
        new_anim = self.animations[self.current_anim_index]
        print(f"▶️ Crossfading to: {new_anim}")
        self.crossfade_to_animation(new_anim, fade_duration=0.5)

        # Schedule the next transition after 5 seconds
        self.taskMgr.doMethodLater(5.0, self.cycle_animations, "CycleAnimationsTask")
        return Task.done

    def start(self):
        # Schedule animation switching every 5 seconds
        self.taskMgr.doMethodLater(5, self.cycle_animations, "CycleAnimationsTask")


if __name__ == '__main__':
    app = App()
    app.run()
