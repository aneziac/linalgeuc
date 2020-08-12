from linalgeuc.graphics import camera, primitive, entity
import pygame as pg
from pygame.locals import *

running = True

view = camera.Viewpoint([900, 600], "rastervfx", "0.3.0")

obj = primitive.Tetrahedron(key='2')
obj.selected = True

while running:

    for event in pg.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False

    entity.Entity.entities.update(pg.key.get_pressed())

    view.render_scene()
