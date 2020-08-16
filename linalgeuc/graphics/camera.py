import linalgeuc.math.linear_algebra as lalib
from linalgeuc.graphics.entity import Entity
from linalgeuc.graphics.mesh import Mesh
from linalgeuc.graphics.primitive import Line
import pygame as pg
from pygame.locals import *
import math
import sys


class Camera(Entity):
    fov_keys = "op"

    def __init__(self, dims=[900, 600], fov=60, pos=[0, 5, 0], key='1', **kwargs):
        self.WIDTH, self.HEIGHT = dims
        self.SCREEN_DIMS = dims

        self.font = pg.font.Font(None, 15)
        self.fov = math.radians(fov)
        self.plane_dist = round((self.WIDTH / 2) / math.tan(self.fov / 2))

        self.label_vertices = False
        self.show_diagnostics = True
        self.show_axes = True
        self.show_edges = True
        self.zoom_speed = 0.1
        self.active = True #False
        self.mode = "perspective"

        if self.show_axes:
            Line([0, 0, 0], [1, 0, 0], color=pg.Color("red"))
            Line([0, 0, 0], [0, 1, 0], color=pg.Color("green"))
            Line([0, 0, 0], [0, 0, 1], color=pg.Color("blue"))

        super().__init__(pos=pos, key=key, **kwargs)

    def update(self, keys, events):
        if self.selected:
            for x in range(len(Camera.fov_keys)):
                if keys[eval("K_" + Camera.fov_keys[x])]:
                    self.fov += ((x * 2) - 1) * 0.01
                    self.plane_dist = round((self.WIDTH / 2) / math.tan(self.fov / 2))
        super().update(keys, events)

    def q1_transform(self, x, y):
        return [x, self.HEIGHT - y]

    def project(self, coord):
        dcoord = self.rotate_point(self.pos - coord, True)

        # cancel if vertex is behind the camera
        if dcoord.y < 0:
            return lalib.InputVector([None] * 2)

        def project_coord(dim, screen_dim):
            if "perspective" in self.mode:
                loc = (self.plane_dist * dcoord.vector[dim]) / (dcoord.y + 10e-5)
            elif "orthographic" in self.mode or "isometric" in self.mode:
                loc = dcoord.vector[dim] * round(self.plane_dist / self.pos.magnitude())

            return math.floor(loc + (screen_dim // 2))

        return lalib.InputVector([project_coord(0, self.WIDTH), project_coord(2, self.HEIGHT)])

    def render_scene(self):
        for entity in Entity.entities.sprites():

            if isinstance(entity, Mesh) and entity.show:
                projected_coords = lalib.Matrix(1, 2)

                for coord in entity.transform().vectors:
                    projected_coords = projected_coords.vcon(self.project(coord).tp())

                if self.show_edges:
                    self.render_edges(entity, projected_coords.matrix)

                if self.label_vertices:
                    self.render_vertex_labels(entity, projected_coords.matrix)

        if self.show_diagnostics:
            self.render_diagnostics()

    def render_edges(self, entity, proj_coords):
        for edge in entity.edges.matrix:
            if len(proj_coords) == entity.vertices.height and None not in proj_coords[edge[0]] + proj_coords[edge[1]]:
                pg.draw.line(self.canvas, entity.color, proj_coords[edge[0]], proj_coords[edge[1]], 3)

    def render_vertex_labels(self, entity, projected_coords):
        for n in range(entity.vertices.height):
            self.text(entity.vertices.get_row(n).round_matrix(2), (projected_coords[n][0] - 20, projected_coords[n][1] + 10), pg.Color("brown"), False)

    def render_diagnostics(self):
        self.text(round(self.clock.get_fps()), [5, self.HEIGHT - 5], after=" FPS")
        self.text(round(math.degrees(self.fov)), [5, self.HEIGHT - 15], after=" FOV")
        self.text(self.pos, [5, self.HEIGHT - 25], after=" POS")
        self.text(self.rot, [5, self.HEIGHT - 35], after=" ROT")
        self.text(len(Entity.entities.sprites()), [5, self.HEIGHT - 45], after=" ENTS")
        self.text(self.mode, [5, self.HEIGHT - 55])

        self.text(self.title, [self.WIDTH - 45, self.HEIGHT - 5])
        self.text(self.version, [self.WIDTH - 45, self.HEIGHT - 15])
        self.text("x".join([str(x) for x in self.SCREEN_DIMS]), [self.WIDTH - 45, self.HEIGHT - 25])

        for entity in Entity.entities.sprites():
            if entity.selected:
                self.text(entity.name + " (" + entity.key + ")", [0, self.HEIGHT - 5], center=True)
                if not isinstance(entity, Camera):
                    self.text(entity.pos, [0, self.HEIGHT - 15], center=True, after=" POS")
                    self.text(entity.rot, [0, self.HEIGHT - 25], center=True, after=" ROT")
                    self.text(entity.scl, [0, self.HEIGHT - 35], center=True, after=" SCL")

    def text(self, text, loc, color=pg.Color("black"), transform=True, after="", center=False):
        if isinstance(text, lalib.Matrix):
            text = str(text.round_matrix(2).make_list())
        if center:
            loc[0] = (self.WIDTH / 2) - (self.font.size(text)[0] / 2)
        if transform:
            self.canvas.blit(self.font.render(str(text) + after, True, color), self.q1_transform(loc[0], loc[1]))
        else:
            self.canvas.blit(self.font.render(str(text) + after, True, color), loc)

    def change_fov(self, x):
        self.fov += ((x * 2) - 1) * 0.01
        self.plane_dist = (self.WIDTH / 2) / math.tan(-self.fov / 2)


class Viewpoint(Camera):
    def __init__(self, dims, title, version="", fill_color=[255] * 3, alpha=False, **kwargs):
        pg.init()
        pg.font.init()

        flags = pg.DOUBLEBUF
        if len(sys.argv) == 1 or "w" not in sys.argv[1]:
            flags = flags | pg.FULLSCREEN | pg.HWSURFACE
        elif len(sys.argv) > 1:
            if "n" in sys.argv[1]:
                flags = flags | pg.NOFRAME
            elif "r" in sys.argv[1]:
                flags = flags | pg.RESIZABLE

        self.WIDTH, self.HEIGHT = dims
        self.fill_color = fill_color
        self.canvas = pg.display.set_mode(dims, flags)
        if not alpha:
            self.canvas.set_alpha(None)
        if version != "":
            version = " v. " + version
        self.title = title
        self.version = version

        pg.display.set_caption(title + version)
        pg.mouse.set_visible(False)
        self.clock = pg.time.Clock()

        super().__init__(**kwargs)

    def update(self, keys, events):

        # allow switching between different projections
        if keys[K_COMMA]:
            self.theta = 0
            self.phi = -math.pi / 2
            self.mode = "front orthographic"
        if keys[K_PERIOD]:
            self.theta = 0
            self.phi = -math.pi
            self.mode = "side orthographic"
        if keys[K_SLASH]:
            self.theta = -math.pi / 2
            self.phi = 0
            self.mode = "top orthographic"
        if keys[K_m]:
            self.theta = -math.pi / 4
            self.phi = -math.pi / 4
            self.mode = "isometric" # FIX LINE BELOW
        if self.mode != "perspective" and self.theta * self.phi != 0 and self.theta + self.phi != 0:
            self.mode = "orthographic"

        if keys[K_b]:
            if self.mode == "perspective":
                self.mode = "orthographic"
            elif self.mode == "orthographic":
                self.mode = "perspective"
            while pg.key.get_pressed()[K_b]:
                pg.event.get()

        self.pantiltzoom(keys, events)
        super().update(keys, events)

    def refresh(self):
        self.clock.tick()
        pg.display.update()
        self.canvas.fill(self.fill_color)

    def reset(self):
        self.theta = math.pi / 6
        self.phi = math.pi / 3
        self.orbit_radius = self.ipos.magnitude()
        self.pan_offset = lalib.InputVector([0, 0, 0])
        super().reset()

    def render_scene(self):
        super().render_scene()
        self.refresh()

    def pantiltzoom(self, keys, events):
        buttons = pg.mouse.get_pressed()
        mpos = pg.mouse.get_pos()

        def mouse_to_value(i, divisor=1000):
            return (mpos[i] - (self.SCREEN_DIMS[i] / 2)) / divisor

        # pan - somewhat buggy
        if keys[K_LSHIFT] and buttons[0]:
            x_dir = lalib.InputVector([1, 0, 0]).rotate_3d(self.rot)
            z_dir = lalib.InputVector([0, 0, 1]).rotate_3d(self.rot)

            self.pan_offset += x_dir.scalar(mouse_to_value(0)) - z_dir.scalar(mouse_to_value(1))

        # zoom
        for x in events:
            if x.type == pg.MOUSEBUTTONDOWN:
                if x.button == 4:
                    self.orbit_radius -= self.zoom_speed
                if x.button == 5:
                    self.orbit_radius += self.zoom_speed

        # tilt / orbit
        if buttons[1]:
            self.phi -= mouse_to_value(0)
            self.theta -= mouse_to_value(1)

        self.pos = lalib.InputVector([self.orbit_radius, self.phi, self.theta]).sphtorect(True) + self.pan_offset
        self.rot = lalib.InputVector([math.degrees(self.theta), 0, math.degrees(-self.phi)])

        pg.mouse.set_pos([self.WIDTH // 2, self.HEIGHT // 2])
