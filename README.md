# linalgeuc

Linalgeuc is a 3D engine for various shapes including cubes, cylinders, and planes that was built using a custom linear algebra library.

# Dependencies

- Python 3
- pygame
- numpy

# Installation and Running

Clone ([or download](https://github.com/aneziac/linalgeuc/archive/master.zip)), install dependencies, and run like the following:

``$ git clone https://github.com/aneziac/linalgeuc.git``

``$ cd linalgeuc``

``$ pip3 install -e .``

``$ python3 -m linalgeuc``

You can also run it windowed with the w flag:

``$ python3 -m linalguec -w``

# Controls

Press 'r' to rotate, 's' to scale, or 't' to translate. 
Then, the axis you affect is controlled by pressing some key in 'qweadx'. 
Revert to initial settings with 'i'. 
Select different entities with the number keys. 
Orbit the camera by holding the middle mouse button and zoom with the scroll wheel.

Press escape to close.

# Credits

Thanks to [@SomeLoserWhoCantPickOriginalUsernames](https://github.com/SomeLoserThatCantPickOriginalUsernames) for originally beginning the linear algebra library.
