import sys
import pygame as pg


class Screen:
    def __init__(self, dims, title, version="", fill_color=[255] * 3, alpha=False):
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

        pg.display.set_caption(title + version)
        pg.mouse.set_visible(False)
        self.clock = pg.time.Clock()

    def update(self):
        self.events = pg.event.get()
        for event in self.events:
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                return False

        self.clock.tick()
        pg.display.update()
        self.canvas.fill(self.fill_color)
        return True
