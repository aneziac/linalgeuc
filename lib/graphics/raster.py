import math
import pygame as pg
import lib.graphics.colors as colors
import lib.math.linear_algebra as lalib


class Entity:
    instances = []

    def __init__(self, pos, rot=[0, 0, 0], scl=[1, 1, 1]):
        Entity.instances.append(self)
        self.ipos = lalib.InputVector(pos)
        self.irot = lalib.InputVector(rot)
        self.iscl = lalib.InputVector(scl).diagonal()
        self.reset()
        self.selected = False
        self.show = True

    def translate_point(self, point):
        return point.add(self.pos.subtract(self.ipos))

    def rotate_point(self, point, reverse=False):
        return point.rotate_3d(self.rot)

    def scale_point(self, point):
        if self.scale_factor < 0:
            self.scale_factor = 0
        self.scl = self.iscl.scalar(self.scale_factor)
        return self.scl.multiply(point)

    def transform_point(self, point):
        return self.translate_point(self.scale_point(self.rotate_point(point)))

    def controls(self, rot_speed, move_speed, scl_speed):
        keys = pg.key.get_pressed()

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
        if keys[pg.K_ESCAPE] or pg.QUIT in [event.type for event in pg.event.get()]:
            import sys
            pg.quit()
            sys.exit()

    def reset(self):
        self.pos = lalib.InputVector(self.ipos.vector)
        self.rot = lalib.InputVector(self.irot.vector)
        self.scale_factor = 1

    @staticmethod
    def select_all():
        for entity in Entity.instances:
            entity.selected = True

    @staticmethod
    def deselect_all():
        for entity in Entity.instances:
            entity.selected = False


class Mesh(Entity):
    def __init__(self, pos, vertices, edges, color, rot=[0, 0, 0], scl=[1, 1, 1]):
        super().__init__(pos, rot, scl)
        self.vertices = vertices
        self.edges = edges
        self.color = color

    def transform(self):
        tvertices = lalib.Matrix(1, 3)
        for x in range(self.vertices.height):
            tvertex = super().transform_point(self.vertices.get_row_as_vec(x))
            tvertices = tvertices.vertical_concatenate(tvertex.transpose())
        return tvertices


class Cube(Mesh):
    def __init__(self, boundaries, color, rot=[0, 0, 0], scl=[1, 1, 1]):
        pos = self.get_pos(boundaries)
        vertices = self.get_vertices(boundaries)
        edges = self.get_edges(vertices)
        super().__init__(pos, vertices, edges, color, rot, scl)

    def get_pos(self, boundaries):
        pos = []
        for x in range(3):
            pos.append((boundaries[0][x] + boundaries[1][x]) / 2)
        return pos

    def get_vertices(self, boundaries):
        def recurse(p, n, c, boundaries):
            for x in c:
                if n == 1:
                    nonlocal vertices
                    vertices = vertices.vertical_concatenate(lalib.InputVector(p + [x]).transpose())
                else:
                    recurse(p + [x], n - 1, [boundaries[0][-n], boundaries[1][-n]], boundaries)

        vertices = lalib.Matrix(1, 3)
        recurse([], len(boundaries[0]), [boundaries[0][0], boundaries[1][0]], boundaries)
        return vertices

    def get_edges(self, vertices):
        edges = lalib.Matrix(1, 2)
        for x in range(vertices.height):
            for y in range(vertices.height):
                xrow = vertices.matrix[x]
                yrow = vertices.matrix[y]
                if x != y and y > x and ((xrow[0] == yrow[0] and xrow[1] == yrow[1]) or
                                         (xrow[1] == yrow[1] and xrow[2] == yrow[2]) or
                                         (xrow[0] == yrow[0] and xrow[2] == yrow[2])):
                    edges = edges.vertical_concatenate(lalib.InputVector([x, y]).transpose())
        return edges


class Tetrahedron(Mesh):
    def __init__(self, center, radius, color, rot=[0, 0, 0], scl=[1, 1, 1]):
        vertices = self.get_vertices(lalib.InputVector(center), radius)
        edges = self.get_edges(vertices)
        super().__init__(center, vertices, edges, color, rot, scl)

    def get_vertices(self, center, radius):
        theta = math.pi / 6
        x = radius * math.cos(theta) * math.cos(theta)
        y = -radius * math.sin(theta) * math.cos(theta)
        z = -radius * math.sin(theta)
        vertices = lalib.InputMatrix([0, 0, radius], [x, y, z], [0, radius * math.cos(theta), z], [-x, y, z])
        return center.stack(4, False).add(vertices)

    def get_edges(self, vertices):
        edges = lalib.Matrix(1, 2)
        for x in range(vertices.height):
            for y in range(vertices.height):
                if x != y and y > x:
                    edges = edges.vertical_concatenate(lalib.InputVector([x, y]).transpose())
        return edges


class Camera(Entity):
    def __init__(self, screen_dims, fov, pos, rot=[0, 0, 0]):
        super().__init__(pos, rot)
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
            return [None, None]
        else:
            return [x, self.screen_dims[1] - y]

    def project(self, coord):
        dcoord = self.rotate_point(self.pos.subtract(coord))

        def projection(dcoord, dim, screen_dim):
            if dcoord.vector[1] > 0:
                return math.floor(((self.plane_dist * dcoord.vector[dim]) / dcoord.vector[1]) + (self.screen_dims[screen_dim] / 2))
            else:
                return None

        return lalib.InputVector(self.q1_transform(projection(dcoord, 0, 0), projection(dcoord, 2, 1)))

    def render_scene(self):
        for entity in Entity.instances:
            if entity.selected:
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

    camera = Camera([900, 600], 55, [0, 3, 0])

    y = Tetrahedron([0, 0, 0], 1, colors.BLACK)
    y.show = False
    x = Cube([[-1, -1, -1], [1, 1, 1]], colors.BLACK)
    x.selected = True

    while True:
        camera.loop()

        keys = pg.key.get_pressed()
        if keys[pg.K_1]:
            Entity.deselect_all()
            camera.selected = True
        if keys[pg.K_2]:
            Entity.deselect_all()
            x.selected = True
        if keys[pg.K_3]:
            Entity.deselect_all()
            y.selected = True


if __name__ == '__main__':
    main()
