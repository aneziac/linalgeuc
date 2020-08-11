import linalgeuc.math.linear_algebra as lalib
import pygame as pg


class Entity:
    instances = []
    tf_keys = "wxadqe"
    scl_keys = "op"
    fov_keys = "op"

    def __init__(self, pos=[0, 0, 0], rot=[0, 0, 0], scl=[1, 1, 1], key=None, name=None):
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
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
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
            self.running = False

    def reset(self):
        if isinstance(self, Camera) and self.ptz:
            self.theta = 0
            self.phi = 0
            self.orbit_radius = self.ipos.distance(lalib.InputVector([0, 0, 0]))
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
