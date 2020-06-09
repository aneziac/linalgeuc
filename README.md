# linalgeuc

Linalgeuc is currently a 3D engine for various shapes including cubes, cylinders, and planes.
However, it is planned to also be a geometry editor and graphing calculator in Euclidean space.
This includes graphing vectors in both 2D and 3D space and animating transformations.

# Dependencies

- Python 3
- pygame
- numpy

# Installation and Running

Clone ([or download](https://github.com/aneziac/linalgeuc/archive/master.zip)), install dependencies, and run like the following:

``$ git clone https://github.com/aneziac/linalgeuc.git``

``$ cd linalgeuc``

``$ pip3 install -e .``

``$ python3 -m linalgeuc`` or ``$ python3 linalgeuc/graphics/raster.py``

# Controls

Press 'r' to rotate, 's' to scale, or 't' to translate. 
Then, the axis you affect is controlled by pressing some key in 'qweadx'. 
Revert to initial settings with 'i'. 
Select different entities with the number keys--note that '1' always controls the camera. 
Orbit the camera by holding the middle mouse button and zoom with the scroll wheel.

Press escape to close.

# Credits

Thanks to [@SomeLoserWhoCantPickOriginalUsernames](https://github.com/SomeLoserThatCantPickOriginalUsernames) for originally beginning the linear algebra library.
