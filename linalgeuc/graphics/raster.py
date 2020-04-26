import math
import pygame as pg
from itertools import permutations
import linalgeuc.graphics.colors as colors
import linalgeuc.math.linear_algebra as lalib


class Entity:
    instances = []

    def __init__(self, pos, key=None, rot=[0, 0, 0], scl=[1, 1, 1]):
        Entity.instances.append(self)
        self.ipos = lalib.InputVector(pos)
        self.irot = lalib.InputVector(rot)
        self.iscl = lalib.InputVector(scl).diagonal()
        self.key = key
        self.reset()
        self.selected = False
        self.show = True

    def translate_point(self, point):
        return point + (self.pos - self.ipos)

    def rotate_point(self, point, reverse=False):
        return point.rotate_3d(self.rot)

    def scale_point(self, point):
        if self.scale_factor < 0:
            self.scale_factor = 0
        self.scl = self.iscl.scalar(self.scale_factor)
        return self.scl * point

    def transform_point(self, point):
        return self.translate_point(self.scale_point(self.rotate_point(point)))

    def controls(self, rot_speed, move_speed, scl_speed):
        keys = pg.key.get_pressed()
        if self.selected:
            if keys[pg.K_q]:
                self.rot.change_item(rot_speed, 2)
            if keys[pg.K_e]:
                self.rot.change_item(-rot_speed, 2)
            if keys[pg.K_a]:
                self.rot.change_item(rot_speed, 1)
            if keys[pg.K_d]:
                self.rot.change_item(-rot_speed, 1)
            if keys[pg.K_s]:
                self.rot.change_item(-rot_speed, 0)
            if keys[pg.K_w]:
                self.rot.change_item(rot_speed, 0)
            if keys[pg.K_LEFT]:
                self.pos.change_item(-move_speed, 0)
            if keys[pg.K_RIGHT]:
                self.pos.change_item(move_speed, 0)
            if keys[pg.K_UP]:
                self.pos.change_item(move_speed, 2)
            if keys[pg.K_DOWN]:
                self.pos.change_item(-move_speed, 2)
            if keys[pg.K_o]:
                self.scale_factor -= scl_speed
            if keys[pg.K_p]:
                self.scale_factor += scl_speed

            if keys[pg.K_r]:
                self.reset()

        if keys[eval("pg.K_" + self.key)]:
            Entity.deselect_all()
            self.selected = True
        if keys[pg.K_ESCAPE] or pg.QUIT in [event.type for event in pg.event.get()]:
            import sys
            pg.quit()
            sys.exit()

    def reset(self):
        self.pos = lalib.InputVector(self.ipos.vector)
        self.rot = lalib.InputVector(self.irot.vector)
        self.scale_factor = 1

    @classmethod
    def select_all(cls):
        for entity in cls.instances:
            entity.selected = True

    @classmethod
    def deselect_all(cls):
        for entity in cls.instances:
            entity.selected = False


class Mesh(Entity):
    def __init__(self, pos, vertices, edges, color, key=None, rot=[0, 0, 0], scl=[1, 1, 1]):
        super().__init__(pos, key, rot, scl)
        self.vertices = vertices
        self.edges = edges
        self.color = color

    def transform(self):
        tvertices = lalib.Matrix(1, 3)
        for x in range(self.vertices.height):
            tvertex = super().transform_point(self.vertices.get_row(x))
            tvertices = tvertices.vertical_concatenate(tvertex.transpose())
        return tvertices


class Plane(Mesh):
    pass


class Circle(Mesh):
    pass


class Polyhedron(Mesh):
    # F + V - E = 2
    pass


class PlatonicSolid(Polyhedron):
    def __init__(self, center, radius, color, key=None, rot=[0, 0, 0], scl=[1, 1, 1]):
        self.golden_ratio = (1 + (5 ** 0.5)) / 2
        self.radius = radius
        vertices = self.get_vertices()
        vertices += lalib.InputVector(center).stack(vertices.height, False)
        edges = self.get_edges(vertices)
        super().__init__(center, vertices, edges, color, key, rot, scl)

    def signs(self, values, permute=False):
        def recurse(n, val, cur, prev=[]):
            for x in cur:
                if n == 1 and [prev + [x]] not in output:
                    output.append(prev + [x])
                else:
                    recurse(n - 1, val, [val[-n]] + [(val[-n] * -1)], prev + [x])

        output = []
        if permute:
            output = list(set(list(permutations(values)) + list(permutations([-x for x in values]))))
        else:
            recurse(len(values), values, [values[0]] + [(values[0] * -1)])
        return lalib.InputMatrix(output)

    def get_edges(self, vertices):
        edges = lalib.Matrix(1, 2)
        v_dist = vertices.get_row(0).magnitude() * 5
        for x in range(vertices.height - 1):
            v_dist = min(v_dist, vertices.get_row(0).distance(vertices.get_row(x + 1)))

        for x in range(vertices.height):
            for y in range(vertices.height):
                if x != y and y > x and vertices.get_row(x).distance(vertices.get_row(y)) == v_dist:
                    edges = edges.vertical_concatenate([x, y])
        return edges


class Tetrahedron(PlatonicSolid):
    def get_vertices(self):
        vertices = super().signs([self.radius] * 2)
        return vertices.horizontal_concatenate(vertices.get_col(0).hadamard_product(vertices.get_col(1)))


class Cube(PlatonicSolid):
    def get_vertices(self):
        return super().signs([self.radius] * 3)


class Octohedron(PlatonicSolid):
    def get_vertices(self):
        return super().signs([0, 0, self.radius], True)


class Dodecahedron(PlatonicSolid):
    def get_vertices(self):
        outer = super().signs([0, self.radius, self.radius / self.golden_ratio], True)
        inner = super().signs([self.radius] * 3)
        return outer.vertical_concatenate(inner)


class Icosahedron(PlatonicSolid):
    def get_vertices(self):
        return super().signs([0, self.radius, self.golden_ratio * self.radius], True)


class Camera(Entity):
    def __init__(self, screen_dims, fov, pos, key=None, rot=[0, 0, 0]):
        super().__init__(pos, key, rot)
        self.screen_dims = screen_dims
        self.screen = pg.display.set_mode(screen_dims)
        self.font = pg.font.Font(None, 15)
        pg.display.set_caption("raster v.0.0.3")
        self.fov = math.radians(fov)
        self.plane_dist = ((screen_dims[0] / 2) / math.tan(fov / 2))
        self.label_vertices = False
        self.show_edges = True

    def q1_transform(self, x, y):
        if x is None or y is None:
            return [None] * 2
        else:
            return [x, self.screen_dims[1] - y]

    def project(self, coord):
        dcoord = self.rotate_point(self.pos - coord)

        def projection(dcoord, dim, screen_dim):
            if dcoord.vector[1] > 0:
                return math.floor(((self.plane_dist * dcoord.vector[dim]) / dcoord.vector[1]) + (self.screen_dims[screen_dim] / 2))
            else:
                return None

        return lalib.InputVector(self.q1_transform(projection(dcoord, 0, 0), projection(dcoord, 2, 1)))

    def render_scene(self):
        for entity in Entity.instances:
            entity.controls(2, 0.01, 0.01)

            if isinstance(entity, Mesh) and entity.show:
                projected_coords = lalib.Matrix(1, 2)

                for coord in entity.transform().matrix:
                    projected_coords = projected_coords.vertical_concatenate(self.project(lalib.InputVector(coord)).transpose())

                if self.show_edges:
                    self.render_edges(entity, projected_coords.matrix)

                if self.label_vertices:
                    self.render_vertex_labels(entity, projected_coords.matrix, self.font)

    def render_edges(self, entity, proj_coords):
        for edge in entity.edges.matrix:
            if len(proj_coords) == entity.vertices.height and None not in proj_coords[edge[0]] + proj_coords[edge[1]]:
                pg.draw.line(self.screen, entity.color, proj_coords[edge[0]], proj_coords[edge[1]], 3)

    def render_vertex_labels(self, entity, projected_coords):
        for n in range(entity.vertices.height):
            text = self.font.render(str(entity.vertices.matrix[n]), True, colors.BROWN)
            self.screen.blit(text, (projected_coords[n][0] - 20, projected_coords[n][1] + 10))

    def loop(self):
        self.screen.fill(colors.WHITE)
        self.render_scene()

        pg.display.update()


def main():
    pg.init()
    pg.font.init()

    camera = Camera([900, 600], 55, [0, 3, 0], '1')

    y = Octohedron([0, 0, 0], 1, colors.BLACK, '3')
    y.show = True
    #x = Cube([0, 0, 0], 1, colors.BLACK, '2')
    y.selected = True

    while True:
        camera.loop()


if __name__ == '__main__':
    main()
