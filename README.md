# linalgeuc

Linalgeuc is a 3D engine for various shapes including cubes, cylinders, and planes that was built using a custom linear algebra library.
It does not claim to follow best practices or be optimized in any way.
Rather, ``linalgeuc`` is intended to be a fun project to learn about the implementation of linear algebra and 3D graphics programming, while trying to rederive as much as possible on my own.

# Installation and Running

Clone ([or download](https://github.com/aneziac/linalgeuc/archive/master.zip)), install dependencies, and run like the following:

``$ git clone https://github.com/aneziac/linalgeuc.git``

``$ cd linalgeuc``

``$ python3 -m linalgeuc``

You can also run it windowed with the w flag:

``$ python3 -m linalguec -w``

# Controls

Press 'r' to rotate, 's' to scale, or 't' to translate.
Then, the axis you affect is controlled by pressing some key in 'qweadx'.
Revert to initial settings with 'i'.
Select different entities with the number keys.
Orbit the viewpoint by holding the middle mouse button, and zoom with the scroll wheel.
Pan by holding left shift and left click.
Change among different projection types by pressing some key in ',./mb'.

Press escape to close.

# Changing the Scene

You can modify what shapes are shown, as well as their properties, by editing the driver code in ``linalgeuc/__main__.py``.
For example, you can change the default icosahedron to a sphere or a cone.

# Supported Shapes

- Lines
- Planes
- Regular Polygons / Circles
- Tetrahedrons
- Cubes
- Octahedrons
- Dodecahedrons
- Icosahedrons
- Prisms / Cylinders
- Pyramids / Cones
- Spheres

[//]: # (Points, Ellipses, and Toruses. Add faces)

# Credits

- Thanks to [@SomeLoserWhoCantPickOriginalUsernames](https://github.com/SomeLoserThatCantPickOriginalUsernames) for originally beginning the linear algebra library.
- Shoutout to [@lama0608](https://github.com/lama0608) for helping test the code.
