from linalgeuc.graphics import camera, primitive, entity
import pygame as pg
from pygame.locals import *

running = True

view = camera.Viewpoint([900, 600], "linalgeuc", "0.4.0")

obj = primitive.Icosahedron(key='2')
obj.selected = True

while running:
    events = pg.event.get()

    for event in events:
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False

    entity.Entity.entities.update(pg.key.get_pressed(), events)

    view.render_scene()
