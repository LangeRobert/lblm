from panda3d.core import GeomVertexFormat, GeomVertexData, Geom, GeomNode, GeomTriangles, GeomVertexWriter
from math import sin, cos, pi

class PieChart3D:
    def __init__(self, parent, radius=1.0, segments=64, color=(0.2, 0.2, 0.2, 0.3)):
        self.radius = radius
        self.segments = segments
        self.color = color
        self.node = parent.attachNewNode("pie_chart")
        self.pie_geom_node = GeomNode("pie_geom")
        self.node.attachNewNode(self.pie_geom_node)

    def update(self, angle_frac: float):  # angle_frac from 0.0 to 1.0
        angle = angle_frac * 2 * pi
        self.pie_geom_node.removeAllGeoms()
        self.pie_geom_node.addGeom(self.create_pie_geom(angle))

    def create_pie_geom(self, angle: float):
        format = GeomVertexFormat.getV3cp()
        vdata = GeomVertexData("vertices", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        color = GeomVertexWriter(vdata, "color")

        vertex.addData3(0, 0, 0)
        color.addData4f(*self.color)

        num_segments = max(1, int(self.segments * (angle / (2 * pi))))
        for i in range(num_segments + 1):
            a = angle * i / num_segments
            x = self.radius * cos(a)
            y = self.radius * sin(a)
            vertex.addData3(x, y, 0)
            color.addData4f(*self.color)

        tris = GeomTriangles(Geom.UHStatic)
        for i in range(1, num_segments):
            tris.addVertices(0, i, i + 1)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        return geom