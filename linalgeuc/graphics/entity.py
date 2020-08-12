import linalgeuc.math.linear_algebra as lalib
import pygame as pg
from pygame.locals import *


class Entity(pg.sprite.Sprite):
    entities = pg.sprite.Group()
    tf_keys = "wxadqe"
    scl_keys = "op"

    def __init__(self, pos=[0, 0, 0], rot=[0, 0, 0], scl=[1, 1, 1], key=None, name=None):
        super().__init__(Entity.entities)

        if not isinstance(pos, lalib.Vector):
            self.ipos = lalib.InputVector(pos)
        else:
            self.ipos = pos

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

    def update(self, keys, rot_speed=2, pos_speed=0.01, scl_speed=0.01):

        if self.selected:
            if keys[pg.K_i]:
                self.reset()

            for tf in {"rot": "r", "pos": "t", "scl": "s"}.items():
                if keys[eval("K_" + tf[1])]:
                    self.selected_tf = tf[0]

            if self.selected_tf is not None:
                if self.selected_tf == "scl":
                    for x in range(len(Entity.scl_keys)):
                        if keys[eval("K_" + Entity.scl_keys[x])]:
                            self.scl = self.scl.scalar(((x * 2) - 1) * scl_speed + 1)

                for action in range(len(Entity.tf_keys)):
                    if keys[eval("K_" + Entity.tf_keys[action])]:
                        op = "self." + self.selected_tf + ".change_item("
                        sign = ("-" if action % 2 == 1 else "")
                        amt = self.selected_tf + "_speed, " + str(action // 2) + ")"
                        eval(op + sign + amt)

            # toggles
            for event in pg.event.get():
                if event.type == KEYDOWN and event.key == K_h:
                    self.show = not self.show

        if self.key is not None and keys[eval("K_" + self.key)]:
            Entity.deselect_all()
            self.selected = True

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

    def reset(self):
        self.pos = self.ipos
        self.rot = self.irot
        self.scl = self.iscl
        self.selected_tf = None

    @classmethod
    def select_all(cls):
        for entity in cls.instances:
            entity.selected = True

    @classmethod
    def deselect_all(cls):
        for entity in cls.instances:
            entity.selected = False
