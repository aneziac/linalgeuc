from linalgeuc.graphics import screen, camera, primitive
import pygame as pg

screen = screen.Screen([900, 600], "rastervfx", "0.2.1")

camera.Camera()

obj = Tetrahedron(key='2')
obj.selected = True

while screen.update():
    camera.render_scene()
