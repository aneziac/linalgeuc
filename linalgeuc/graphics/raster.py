import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import math
import pygame as pg
from collections import deque
import linalgeuc.graphics.colors as colors
import linalgeuc.math.linear_algebra as lalib


class Entity:
    instances = []
    tf_keys = "wxadqe"
    scl_keys = "op"
    fov_keys = "op"

    def __init__(self, pos=[0, 0, 0], rot=[0, 0, 0], scl=[1, 1, 1], key=None):
        Entity.instances.append(self)

        if not isinstance(pos, lalib.Matrix):
            self.ipos = lalib.InputVector(pos)
        else:
            self.ipos = pos

        if isinstance(self, Mesh):
            self.vertices += self.ipos.stack(self.vertices.height, False)

        self.irot = lalib.InputVector(rot)
        self.iscl = lalib.InputVector(scl)
        self.key = key
        self.reset()
        self.selected = False
        self.show = True

    def translate_point(self, point):
        return point + (self.pos - self.ipos)

    def rotate_point(self, point, reverse=False):
        if reverse:
            return point.rotate_3d(-self.rot)
        else:
            return point.rotate_3d(self.rot)

    def scale_point(self, point):
        for x in range(self.scl.height):
            self.scl.vector[x] = max(0, self.scl.vector[x])
        return self.iscl.hadp(self.scl).hadp(point)

    def transform_point(self, point):
        return self.translate_point(self.scale_point(self.rotate_point(point)))

    def controls(self, rot_speed=2, pos_speed=0.01, scl_speed=0.01):
        keys = pg.key.get_pressed()
        events = [event for event in pg.event.get()]

        if isinstance(self, Camera) and self.ptz:
            self.pantiltzoom(keys, events)

        if self.selected:
            if keys[pg.K_i]:
                self.reset()

            if isinstance(self, Camera):
                for x in range(len(Entity.fov_keys)):
                    if keys[eval("pg.K_" + Entity.fov_keys[x])]:
                        self.fov += ((x * 2) - 1) * 0.01

            for tf in {"rot": "r", "pos": "t", "scl": "s"}.items():
                if keys[eval("pg.K_" + tf[1])]:
                    self.selected_tf = tf[0]

            if self.selected_tf is not None:
                if self.selected_tf == "scl":
                    for x in range(len(Entity.scl_keys)):
                        if keys[eval("pg.K_" + Entity.scl_keys[x])]:
                            self.scl = self.scl.scalar(((x * 2) - 1) * scl_speed + 1)

                for action in range(len(Entity.tf_keys)):
                    if keys[eval("pg.K_" + Entity.tf_keys[action])]:
                        op = "self." + self.selected_tf + ".change_item("
                        amt = ("-" if action % 2 == 1 else "") + self.selected_tf + "_speed, " + str(action // 2) + ")"
                        eval(op + amt)

            if keys[pg.K_h]:
                while pg.key.get_pressed()[pg.K_h]:
                    pg.event.get()
                self.show = not self.show

        if self.key is not None and keys[eval("pg.K_" + self.key)]:
            Entity.deselect_all()
            self.selected = True
        if keys[pg.K_ESCAPE] or pg.QUIT in [x.type for x in events]:
            import sys
            pg.quit()
            sys.exit()

    def reset(self):
        if isinstance(self, Camera) and self.ptz:
            self.theta = 0
            self.phi = 0
            self.rad = self.ipos.distance(lalib.InputVector([0, 0, 0]))
        self.pos = lalib.InputVector(self.ipos.vector)
        self.rot = lalib.InputVector(self.irot.vector)
        self.scl = lalib.InputVector(self.iscl.vector)
        self.selected_tf = None

    @classmethod
    def select_all(cls):
        for entity in cls.instances:
            entity.selected = True

    @classmethod
    def deselect_all(cls):
        for entity in cls.instances:
            entity.selected = False


class Mesh(Entity):
    def __init__(self, vertices, edges, color=colors.BLACK, **kwargs):
        self.vertices = vertices
        self.edges = edges
        self.color = color
        super().__init__(**kwargs)

    def transform(self):
        tvertices = lalib.Matrix(1, 3)
        for x in range(self.vertices.height):
            tvertex = super().transform_point(self.vertices.get_row(x))
            tvertices = tvertices.vcon(tvertex.tp())
        return tvertices


class Line(Mesh):
    def __init__(self, start=[-1, 0, 0], end=[1, 0, 0], **kwargs):
        start, end = lalib.InputVector(start), lalib.InputVector(end)
        vertices = start.tp().vcon(end.tp())
        super().__init__(vertices, lalib.InputMatrix([[0, 1]]), pos=start.midpoint(end), **kwargs)


class Regular(Mesh):
    def __init__(self, radius=1, **kwargs):
        self.radius = radius
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
        self.resolution = resolution
        self.height = height
        super().__init__(**kwargs)

    def approx_circle(self, height=0):
        vertices = lalib.Matrix(1, 3)
        angle_inc = (2 * math.pi) / self.resolution

        for x in range(self.resolution):
            vertices = vertices.vcon(lalib.InputVector([math.cos(angle_inc * x), math.sin(angle_inc * x), height]).tp())

        return vertices


# POLYGONS

class Plane(Regular):
    def get_vertices(self):
        return super().signs([self.radius] * 2).hcon(lalib.Matrix.zeros(4, 1))


class Circle(Circular):
    def get_vertices(self):
        return super().approx_circle()


# POLYHEDRA

class PlatonicSolid(Regular):
    def __init__(self, **kwargs):
        self.golden_ratio = (1 + (5 ** 0.5)) / 2
        super().__init__(**kwargs)


class Tetrahedron(PlatonicSolid):
    def get_vertices(self):
        vertices = super().signs([self.radius] * 2)
        return vertices.hcon(vertices.get_col(0).hadp(vertices.get_col(1)))


class Cube(PlatonicSolid):
    def get_vertices(self):
        return super().signs([self.radius] * 3)


class Octahedron(PlatonicSolid):
    def get_vertices(self):
        return super().signs([0, 0, self.radius])


class Dodecahedron(PlatonicSolid):
    def get_vertices(self):
        outer = super().signs([0, self.radius * self.golden_ratio, self.radius / self.golden_ratio])
        inner = super().signs([self.radius] * 3)
        return outer.vcon(inner)


class Icosahedron(PlatonicSolid):
    def get_vertices(self):
        return super().signs([0, self.radius, self.golden_ratio * self.radius])


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
    pass


# OTHER ENTITIES

class Empty(Entity):
    pass


class Camera(Entity):
    def __init__(self, screen_dims=[900, 600], fov=60, pos=[0, 5, 0], rot=[0, 0, 0], key='1', **kwargs):
        self.screen_dims = screen_dims
        self.screen = pg.display.set_mode(screen_dims)
        self.font = pg.font.Font(None, 15)
        self.fov = math.radians(fov)
        self.clock = pg.time.Clock()

        self.title = "rasterfx"
        self.version = "v.0.2.1"
        pg.display.set_caption(self.title + " " + self.version)

        self.label_vertices = False
        self.show_diagnostics = True
        self.show_edges = True
        self.ptz = True

        super().__init__(pos=pos, rot=rot, key=key, **kwargs)

    def q1_transform(self, x, y):
        if x is None or y is None:
            return [None] * 2
        else:
            return [x, self.screen_dims[1] - y]

    def project(self, coord):
        dcoord = self.rotate_point(self.pos - coord, True)

        def projection(dcoord, dim, screen_dim):
            if dcoord.vector[1] > 0:
                return math.floor(((self.plane_dist * dcoord.vector[dim]) / dcoord.vector[1]) + (self.screen_dims[screen_dim] / 2))
            else:
                return None

        return lalib.InputVector(self.q1_transform(projection(dcoord, 0, 0), projection(dcoord, 2, 1)))

    def render_scene(self):
        self.plane_dist = ((self.screen_dims[0] / 2) / math.tan(-self.fov / 2))

        for entity in Entity.instances:
            entity.controls()

            if isinstance(entity, Mesh) and entity.show:
                projected_coords = lalib.Matrix(1, 2)

                for coord in entity.transform().matrix:
                    projected_coords = projected_coords.vcon(self.project(lalib.InputVector(coord)).tp())

                if self.show_edges:
                    self.render_edges(entity, projected_coords.matrix)

                if self.label_vertices:
                    self.render_vertex_labels(entity, projected_coords.matrix)

        if self.show_diagnostics:
            self.render_diagnostics()

        self.clock.tick()

    def render_edges(self, entity, proj_coords):
        for edge in entity.edges.matrix:
            if len(proj_coords) == entity.vertices.height and None not in proj_coords[edge[0]] + proj_coords[edge[1]]:
                pg.draw.line(self.screen, entity.color, proj_coords[edge[0]], proj_coords[edge[1]], 3)

    def render_vertex_labels(self, entity, projected_coords):
        for n in range(entity.vertices.height):
            self.text(entity.vertices.get_row(n).round_matrix(2), (projected_coords[n][0] - 20, projected_coords[n][1] + 10), colors.BROWN, False)

    def render_diagnostics(self):
        self.text(round(self.clock.get_fps()), [5, self.screen_dims[1] - 5], after=" FPS")
        self.text(round(math.degrees(self.fov)), [5, self.screen_dims[1] - 15], after=" FOV")
        self.text(self.pos, [5, self.screen_dims[1] - 25], after=" POS")
        self.text(self.rot, [5, self.screen_dims[1] - 35], after=" ROT")
        self.text(len(Entity.instances), [5, self.screen_dims[1] - 45], after=" ENTS")

        self.text(self.title, [self.screen_dims[0] - 45, self.screen_dims[1] - 5])
        self.text(self.version, [self.screen_dims[0] - 45, self.screen_dims[1] - 15])
        self.text("x".join([str(x) for x in self.screen_dims]), [self.screen_dims[0] - 45, self.screen_dims[1] - 25])

        for entity in Entity.instances:
            if entity.selected:
                self.text(entity.__class__.__name__ + " (" + entity.key + ")", [0, self.screen_dims[1] - 5], center=True)
                if not isinstance(entity, Camera):
                    self.text(entity.pos, [0, self.screen_dims[1] - 15], center=True, after=" POS")
                    self.text(entity.rot, [0, self.screen_dims[1] - 25], center=True, after=" ROT")
                    self.text(entity.scl, [0, self.screen_dims[1] - 35], center=True, after=" SCL")

    def text(self, text, loc, color=colors.BLACK, transform=True, after="", center=False):
        if isinstance(text, lalib.Matrix):
            text = str(text.round_matrix(2).make_list())
        if center:
            loc[0] = (self.screen_dims[0] / 2) - (self.font.size(text)[0] / 2)
        if transform:
            self.screen.blit(self.font.render(str(text) + after, True, color), self.q1_transform(loc[0], loc[1]))
        else:
            self.screen.blit(self.font.render(str(text) + after, True, color), loc)

    def convang(self, i):
        mpos = pg.mouse.get_pos()
        return (mpos[i] - (self.screen_dims[i] / 2)) / 1000

    def pantiltzoom(self, keys, events):
        buttons = pg.mouse.get_pressed()

        # zoom
        for x in events:
            if x.type == pg.MOUSEBUTTONDOWN:
                if x.button == 4:
                    self.rad -= 0.1
                if x.button == 5:
                    self.rad += 0.1

        # tilt / orbit
        if buttons[1]:
            self.phi -= self.convang(0)
            self.theta += self.convang(1)

        self.pos = lalib.InputVector([self.rad, self.phi, self.theta]).sphtorect(True)
        self.rot = lalib.InputVector([math.degrees(self.theta), 0, math.degrees(-self.phi)])

        pg.mouse.set_visible(False)
        pg.mouse.set_pos([self.screen_dims[0] / 2, self.screen_dims[1] / 2])

    def loop(self):
        while True:
            self.screen.fill(colors.WHITE)
            self.render_scene()

            pg.display.update()


def main():
    pg.init()
    pg.font.init()

    camera = Camera()

    y = Cylinder(key='2')
    y.selected = True

    camera.loop()


if __name__ == '__main__':
    main()
