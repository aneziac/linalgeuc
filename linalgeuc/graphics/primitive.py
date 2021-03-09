import linalgeuc.math.linear_algebra as lalib
from linalgeuc.graphics.mesh import Mesh, Regular, Circular
import math

# CONSTANTS

GOLDEN_RATIO = (1 + (5 ** 0.5)) / 2


# 0D

class Point(Mesh):
    pass


# 1D

class Line(Mesh):
    def __init__(self, start=[-1, 0, 0], end=[1, 0, 0], **kwargs):
        start, end = lalib.InputVector(start), lalib.InputVector(end)
        vertices = start.tp().vcon(end.tp())
        super().__init__(vertices, lalib.InputMatrix([[0, 1]]), pos=start.midpoint(end), add_pos=False, **kwargs)


# 2D

class Plane(Regular):
    def get_vertices(self):
        return super().signs([self.radius] * 2).hcon(lalib.Matrix.zeros(4, 1))


class Circle(Circular):
    def get_vertices(self):
        return super().approx_circle()


# 3D

## Platonic Solids

class Tetrahedron(Regular):
    def get_vertices(self):
        vertices = super().signs([self.radius] * 2)
        return vertices.hcon(vertices.get_col(0).hadp(vertices.get_col(1)))


class Cube(Regular):
    def get_vertices(self):
        return super().signs([self.radius] * 3)


class Octahedron(Regular):
    def get_vertices(self):
        return super().signs([0, 0, self.radius])


class Dodecahedron(Regular):
    def get_vertices(self):
        outer = super().signs([0, self.radius * GOLDEN_RATIO, self.radius / GOLDEN_RATIO])
        inner = super().signs([self.radius] * 3)
        return outer.vcon(inner)


class Icosahedron(Regular):
    def get_vertices(self):
        return super().signs([0, self.radius, GOLDEN_RATIO * self.radius])


## Circular Shapes

class Cylinder(Circular):
    def get_vertices(self):
        top_half = super().approx_circle(self.height / 2)
        return super().approx_circle(-self.height / 2).vcon(top_half)

    def get_edges(self, vertices):
        edges = super().get_edges(vertices)
        for x in range(vertices.height):
            for y in range(vertices.height):
                if vertices.get_row(x).add(lalib.InputVector([0, 0, self.height])) == vertices.get_row(y):
                    edges = edges.vcon(lalib.InputVector([x, y]))
        return edges


class Cone(Circular):
    def get_vertices(self):
        top = lalib.InputVector([0, 0, self.height / 2]).tp()
        return top.vcon(super().approx_circle(-self.height / 2))

    def get_edges(self, vertices):
        edges = super().get_edges(vertices)
        for x in range(vertices.height - 1):
            edges = edges.vcon(lalib.InputVector([0, x + 1]))
        return edges


class Sphere(Circular):
    def __init__(self, vresolution=5, hresolution=6, **kwargs):
        assert vresolution >= 3
        assert hresolution >= 3
        self.vresolution = vresolution
        self.ring_count = self.vresolution - 2
        self.hresolution = hresolution
        super().__init__(hresolution, **kwargs)
        self.ring_spacing = 2 * self.radius / self.vresolution

    def get_vertices(self):
        top = lalib.InputVector([0, 0, self.radius]).tp()
        bottom = lalib.InputVector([0, 0, -self.radius]).tp()
        for x in range(self.ring_count):
            angle = (math.pi / (self.vresolution - 1)) * (x + 1)
            height = self.radius * math.cos(angle)
            radius = self.radius * math.sin(angle)
            top = top.vcon(super().approx_circle(height, radius))
        return top.vcon(bottom)

    def get_edges(self, vertices):
        edges = lalib.Matrix(1, 2)
        vnum = (self.hresolution * self.ring_count) + 2

        # north pole
        for x in range(self.hresolution):
            edges = edges.vcon(lalib.InputVector([0, x + 1]).tp())

        # longitude
        for x in range(self.hresolution * (self.ring_count - 1)):
            edges = edges.vcon(lalib.InputVector([x + 1, x + 1 + self.hresolution]))

        # latitude
        for r in range(self.ring_count):
            layer = (r * self.hresolution) + 1
            for h in range(self.hresolution - 1):
                edges = edges.vcon(lalib.InputVector([layer + h, layer + h + 1]))
            edges = edges.vcon(lalib.InputVector([layer, (r + 1) * self.hresolution]))

        # south pole
        for x in range(self.hresolution):
            edges = edges.vcon(lalib.InputVector([vnum - 2 - x, vnum - 1]))

        return edges


class Torus(Circular):  # work in progress
    def __init__(self, minor_radius=0.5, major_radius=1, vresolution=5, hresolution=6, **kwargs):
        assert major_radius > minor_radius > 0
        self.minor_radius = minor_radius
        self.major_radius = major_radius
        assert vresolution >= 3
        assert hresolution >= 3
        self.vresolution = vresolution
        self.ring_count = self.vresolution - 2
        self.hresolution = hresolution

    def get_vertices(self):
        bottom = super().approx_circle(-self.height / 2, (self.minor_radius + self.major_radius) / 2)
        for x in range(self.ring_count):
            angle = (math.pi / (self.vresolution - 1)) * (x + 1)
            height = self.major_radius * math.sin(angle) * self.height
            bottom.vcon(super().approx_circle(height, self.minor_radius))
            bottom.vcon(super().approx_circle(height, self.major_radius))

        bottom.vcon(super().approx_circle(-self.height / 2, (self.minor_radius + self.major_radius) / 2))
        return bottom

    def get_edges(self):
        # longitude
        for x in range(self.hresolution * (self.ring_count - 1)):
            edges = edges.vcon(lalib.InputVector([x + 1, x + 1 + self.hresolution]))

        # latitude
        for r in range(self.ring_count):
            layer = (r * self.hresolution) + 1
            for h in range(self.hresolution - 1):
                edges = edges.vcon(lalib.InputVector([layer + h, layer + h + 1]))
            edges = edges.vcon(lalib.InputVector([layer, (r + 1) * self.hresolution]))
