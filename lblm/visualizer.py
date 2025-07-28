import time
from multiprocessing.synchronize import Event

from panda3d.core import loadPrcFileData, Shader
loadPrcFileData('', 'basic-shaders-only #f')  # Allow custom shaders

from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, Vec4
from direct.actor.Actor import Actor
from direct.task import Task
from direct.interval.LerpInterval import LerpFunc
from multiprocessing import Process, Queue
import sys

from panda3d.core import GeomVertexFormat, GeomVertexData, Geom, GeomNode, GeomTriangles, GeomVertexWriter
from math import sin, cos, pi

class Ring3D:
    def __init__(self, parent, radius=1.0, segments=64, color=(1.0, 1.0, 1.0, 1.0), thickness=0.05):
        self.radius = radius
        self.segments = segments
        self.color = color
        self.thickness = thickness
        self.node = parent.attachNewNode("ring")
        self.ring_geom_node = GeomNode("ring_geom")
        self.node.attachNewNode(self.ring_geom_node)
        self.is_visible = True
        self._create_ring()

    def update(self, visible: bool):
        self.is_visible = visible
        if visible:
            self.node.show()
        else:
            self.node.hide()

    def _create_ring(self):
        format = GeomVertexFormat.getV3cp()
        vdata = GeomVertexData("vertices", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")

        inner_radius = self.radius - self.thickness / 2
        outer_radius = self.radius + self.thickness / 2

        # Create vertices for inner and outer circles
        for i in range(self.segments):
            angle = 2 * pi * i / self.segments

            # Inner circle vertex
            x_inner = inner_radius * cos(angle)
            y_inner = inner_radius * sin(angle)
            vertex.addData3(x_inner, y_inner, 0)
            color.addData4f(*self.color)

            # Outer circle vertex
            x_outer = outer_radius * cos(angle)
            y_outer = outer_radius * sin(angle)
            vertex.addData3(x_outer, y_outer, 0)
            color.addData4f(*self.color)

        # Create triangles for the ring
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(self.segments):
            next_i = (i + 1) % self.segments

            # Two triangles per segment
            # Triangle 1: inner[i], outer[i], inner[next_i]
            tris.addVertices(i * 2, i * 2 + 1, next_i * 2)

            # Triangle 2: inner[next_i], outer[i], outer[next_i]
            tris.addVertices(next_i * 2, i * 2 + 1, next_i * 2 + 1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        self.ring_geom_node.addGeom(geom)


class Visualizer(Process):

    def __init__(self,
                 options,
                 queue: Queue,
                 is_outputting_event: Event,
                 ):
        super().__init__()
        self.options = options
        self.queue = queue
        self.is_outputting_event = is_outputting_event

    def run(self):
        try:
            vis = _Visualizer(self.options, self.queue, self.is_outputting_event)
            vis.start()
        except Exception as e:
            print(f"Error in visualizer process: {e}")
            import traceback
            traceback.print_exc()


class _Visualizer(ShowBase):
    def __init__(self, options, queue: Queue, is_outputting_event: Event, actor_path="lblm/data/character.glb"):
        try:
            ShowBase.__init__(self)
            self.queue = queue
            self.is_outputting_event = is_outputting_event

            # init the animations and the base animation
            self.current_anim_index = 0
            print(f"Available animations: {list(options.keys())}")

            self.disableMouse()
            self.set_background_color(0, 0, 0, 1)

            # Load GLB model with rig
            self.animations = options

            self.actor = Actor(actor_path, self.animations)
            self.actor.reparentTo(self.render)
            self.actor.setScale(1)
            self.actor.setPos(0, 0, 0)

            # animations
            self.ring = Ring3D(self.render)
            self.pie_duration = 8.0  # seconds for full animation
            self.taskMgr.add(self.update_animations, "update_animations")
            self.animating = False

            self.animation_length = 0
            self.animation_start = 0

            # Load and apply the shader
            try:
                shader = Shader.load(Shader.SL_GLSL, "lblm/shader/gradient.vert", "lblm/shader/gradient.frag")
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
                cam_dist = 3  # How far in front of the model
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
                self.is_outputting_event.clear()
            else:
                value = self.queue.get(block=False)
                if value and value in self.animations:
                    to_anim_index = list(self.animations.keys()).index(value)
                    self.is_outputting_event.set()
                else:
                    to_anim_index = 0
                    self.is_outputting_event.clear()

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
            self.animation_length = anim_length
            self.animation_start = time.time()
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

    def update_animations(self, task):
        if not self.is_outputting_event.is_set():
            # show the pie chart
            elapsed = int(self.animation_start - time.time())

            # hide the ring
            self.ring.update(False)
        else:
            self.ring.update(True)
        return Task.cont
