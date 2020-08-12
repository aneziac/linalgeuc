import linalgeuc.math.linear_algebra as lalib
from linalgeuc.graphics.entity import Entity
import pygame as pg
import math
from collections import deque


class Mesh(Entity):
    def __init__(self, vertices, edges, color=pg.Color("black"), **kwargs):
        self.vertices = vertices
        self.edges = edges
        self.color = color

        self.vertex_amt = self.vertices.size[0]
        self.edges_amt = self.edges.size[0]
        super().__init__(**kwargs)
        self.vertices += self.ipos.stack(self.vertices.height, False)

    def transform(self):
        tvertices = lalib.Matrix(1, 3)
        for x in range(self.vertices.height):
            tvertex = super().transform_point(self.vertices.get_row(x))
            tvertices = tvertices.vcon(tvertex.tp())
        return tvertices


class Regular(Mesh):
    def __init__(self, radius=1, **kwargs):
        self.radius = radius
        self.diameter = 2 * self.radius
        vertices = self.get_vertices()
        edges = self.get_edges(vertices)
        super().__init__(vertices, edges, **kwargs)

    def signs(self, values):
        def recurse(n, val, cur=None, prev=[]):
            if cur is None:
                recurse(n, val, [val[0]] + [(val[0] * -1)])
            else:
                for x in cur:
                    if n == 1 and prev + [x] not in output:
                        output.append(prev + [x])
                    elif n > 1:
                        recurse(n - 1, val, [val[-n + 1]] + [(val[-n + 1] * -1)], prev + [x])

        output = []
        lst = deque(values)
        for x in range(len(values)):
            recurse(len(values), list(lst))
            lst.rotate()

        return lalib.InputMatrix(output)

    def get_edges(self, vertices):
        edges = lalib.Matrix(1, 2)
        v_dist = vertices.get_row(vertices.height - 2).distance(vertices.get_row(vertices.height - 1))

        for x in range(vertices.height - 3):
            v_dist = min(v_dist, vertices.get_row(vertices.height - 2).distance(vertices.get_row(vertices.height - x - 3)))

        for x in range(vertices.height):
            for y in range(vertices.height):
                if x != y and y > x and round(vertices.get_row(x).distance(vertices.get_row(y)), 2) <= round(v_dist, 2):
                    edges = edges.vcon([x, y])
        return edges


class Circular(Regular):
    def __init__(self, resolution=16, height=2, **kwargs):
        assert resolution >= 3
        self.resolution = resolution
        self.height = height
        super().__init__(**kwargs)

    def approx_circle(self, height=0, radius=1): # swap parameters
        vertices = lalib.Matrix(1, 3)
        angle_inc = (2 * math.pi) / self.resolution

        for x in range(self.resolution):
            vertices = vertices.vcon(lalib.InputVector([math.cos(angle_inc * x) * radius, math.sin(angle_inc * x) * radius, height]).tp())

        return vertices
