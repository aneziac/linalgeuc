import linalgeuc.math.linear_algebra as lalib
import pygame as pg

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
        super().__init__(vertices, lalib.InputMatrix([[0, 1]]), pos=start.midpoint(end), **kwargs)


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
    def __init__(self, vresolution=5, hresolution=8, **kwargs):
        assert vresolution >= 3
        assert hresolution >= 3
        self.vresolution = vresolution
        self.hresolution = hresolution
        super().__init__(hresolution, **kwargs)

    def get_vertices(self):
        bottom = lalib.InputVector([0, 0, -self.radius])
        top = lalib.InputVector([0, 0, self.radius])
        for x in range(self.vresolution - 2):
            h = self.radius * ((x / (self.vresolution - 2) * 2) - 1)
            r = math.sin(math.pi * (x + 1) / self.vresolution)
            top = top.vcon(super().approx_circle(h, r).tp())
        return top.vcon(bottom)

    def get_edges(self):
        edges = lalib.Vector(2)

        # south pole
        for x in range(self.hresolution):
            edges = edges.vcon(lalib.InputVector([0, x + 1]))

        # longitude
        vnum = self.hresolution * (self.vresolution - 2)
        for x in range(vnum - 2):
            edges = edges.vcon(lalib.InputVector([x + 1, x + 1 + self.hresolution]))

        # latitude
        for v in range(self.vresolution - 2):
            for h in range(self.hresolution - 1):
                edges = edges.vcon(lalib.InputVector([(v * self.hresolution) + h + 1, h + 2]))

        # north pole
        for x in range(self.hresolution):
            edges = edges.vcon(lalib.InputVector([vnum - 2 - x, vnum - 1]))

        return edges
