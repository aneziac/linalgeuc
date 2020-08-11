import linalgeuc.math.linear_algebra as lalib
import pygame as pg
import math


class Camera(Entity): # viewpoint that inherits?
    def __init__(self, dims=[900, 600], fullscreen=False, fov=60, pos=[0, 5, 0], key='1', **kwargs):
        self.WIDTH, self.HEIGHT = dims

        self.font = pg.font.Font(None, 15)
        self.fov = math.radians(fov)
        self.running = True

        self.label_vertices = False
        self.show_diagnostics = True
        self.show_edges = True
        self.ptz = True
        self.zoom_speed = 0.1

        super().__init__(pos=pos, key=key, **kwargs)

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
                self.text(entity.name + " (" + entity.key + ")", [0, self.screen_dims[1] - 5], center=True)
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
                    self.orbit_radius -= self.zoom_speed
                if x.button == 5:
                    self.orbit_radius += self.zoom_speed

        # tilt / orbit
        if buttons[1]:
            self.phi -= self.convang(0)
            self.theta += self.convang(1)

        self.pos = lalib.InputVector([self.orbit_radius, self.phi, self.theta]).sphtorect(True)
        self.rot = lalib.InputVector([math.degrees(self.theta), 0, math.degrees(-self.phi)])

        pg.mouse.set_visible(False)
        pg.mouse.set_pos([self.screen_dims[0] / 2, self.screen_dims[1] / 2])

    def loop(self):
        while self.running:
            self.screen.fill(colors.WHITE)
            self.render_scene()

            pg.display.update()
