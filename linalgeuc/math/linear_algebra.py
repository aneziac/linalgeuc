import math
import random
import numpy as np


class Matrix:
    def __init__(self, height, width):
        self.height = height
        self.rows = height
        self.width = width
        self.columns = width
        self.clear()

    def clear(self):
        self.matrix = [[None for y in range(self.width)] for x in range(self.height)]
        if isinstance(self, Vector):
            self.vector = [None for x in range(self.height)]
        self.cursor = 0

    def push(self, item):
        def push_single(item):
            try:
                if isinstance(self, Vector):
                    self.vector[self.cursor] = item
                    if self.height <= 3:
                        self.x, self.y, self.z = self.vector + [None] * (3 - self.height)
                self.matrix[self.cursor // self.width][self.cursor % self.width] = item
                self.cursor += 1

            except IndexError:
                raise ValueError("Matrix full")

        if isinstance(item, list):
            for x in item:
                push_single(x)
        else:
            push_single(item)

    def print(self, round_vals=False, name=None):
        if name is not None:
            print(name + ' =')
        for x in range(self.height):
            for y in range(self.width):
                if round_vals:
                    print(round(self.matrix[x][y]), end=' ')
                else:
                    print(self.matrix[x][y], end=' ')
            print(" ")
        print(" ")

    @property
    def size(self):
        return [self.height, self.width]

    def set_item(self, item, row, col=0):
        if isinstance(self, Vector):
            self.vector[row] = item
        self.matrix[row][col] = item

    __setitem__ = set_item

    def change_item(self, delta, row, col=0):
        if isinstance(self, Vector):
            self.vector[row] += delta
        self.matrix[row][col] += delta

    @property
    def num_items(self):
        return self.height * self.width

    def get_item(self, row, col):
        return self.matrix[row][col]

    __getitem__ = get_item

    def get_column_as_list(self, x):
        return [row[x] for row in self.matrix]

    def get_column(self, x):
        return InputVector(self.get_column_as_list(x))

    get_col = get_column

    def get_row_as_list(self, y):
        return self.matrix[y]

    def get_row(self, y):
        return InputVector(self.get_row_as_list(y))

    def make_list(self):
        if isinstance(self, Vector):
            return self.vector
        else:
            return self.matrix

    @property
    def vectors(self):
        result = []        

        for row in self.matrix:
            result.append(InputVector(row))
        return result

    def ensure_equal_heights(self, other_matrix, dim="heights"):
        if self.height != other_matrix.height:
            raise ValueError("Corresponding matrix " + dim + " must be equal (" + str(self.height) + " =/= " + str(other_matrix.height) + ")")
        return True

    def ensure_equal_widths(self, other_matrix):
        return self.transpose().ensure_equal_heights(other_matrix.transpose(), "widths")

    def ensure_equal_dims(self, other_matrix):
        self.ensure_full()
        self.ensure_equal_heights(other_matrix)
        self.ensure_equal_widths(other_matrix)
        return True

    def ensure_square(self):
        self.ensure_full()
        if self.height != self.width:
            raise ValueError("Matrix must be square")
        return True

    def ensure_vector(self):
        self.ensure_full()
        if not isinstance(self, Vector):
            raise ValueError("Must input vector")
        return True

    def ensure_full(self):
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] is None:
                    raise ValueError("At least some elements of the matrix are empty")
        return True

    @property
    def has_values(self):
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] is not None:
                    return True
        return False

    def is_in(self, val):
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] == val:
                    return True
        return False

    def is_same(self, other_matrix):
        if self.matrix == other_matrix.matrix:
            return True
        return False

    @classmethod
    def identity(cls, n):
        result = cls(n, n)
        for x in range(n):
            for y in range(n):
                if x == y:
                    result.push(1)
                else:
                    result.push(0)
        return result

    @staticmethod
    def ones(n, m=None):
        if m is None:
            m = n
        result = create_array(n, m)
        for x in range(n):
            result.push([1 for y in range(m)])
        return result

    @staticmethod
    def zeros(n, m=None):
        if m is None:
            m = n
        result = create_array(n, m)
        for x in range(n):
            result.push([0 for y in range(m)])
        return result

    def transpose(self):
        result = create_array(self.width, self.height)
        for x in range(self.width):
            result.push([self.matrix[y][x] for y in range(self.height)])
        return result

    tp = transpose

    def add(self, other_matrix):
        self.ensure_equal_dims(other_matrix)

        result = create_array(self.height, self.width)
        for x in range(self.height):
            result.push([(self.matrix[x][y] + other_matrix.matrix[x][y]) for y in range(self.width)])
        return result

    __add__ = add

    def subtract(self, other_matrix):
        return self + other_matrix.scalar(-1)

    __sub__ = subtract

    def submatrix(self, start_row, start_col, end_row, end_col):
        assert start_row >= 0
        assert start_col >= 0
        assert end_row <= self.height
        assert end_col <= self.width

        result = Matrix(end_row - start_row + 1, end_col - start_col + 1)
        for x in range(self.height):
            for y in range(self.width):
                if end_row >= x >= start_row and end_col >= y >= start_col:
                    result.push(self.matrix[x][y])
        return result

    def minor(self, row, col):
        if self.height != self.width or self.height <= 2:
            raise ValueError("Invalid minor")

        result = Matrix(self.height - 1, self.width - 1)
        for x in range(self.height):
            for y in range(self.width):
                if x != row and y != col:
                    result.push(self.matrix[x][y])
        return result.det

    @property
    def determinant(self):
        self.ensure_square()

        if self.height == 2:
            return self.matrix[0][0] * self.matrix[1][1] - (self.matrix[0][1] * self.matrix[1][0])

        sum = 0
        for y in range(self.width):
            sum += self.matrix[0][y] * self.minor(0, y) * math.cos(y * math.pi)
        return sum

    det = determinant

    def scalar(self, scl):
        result = create_array(self.height, self.width)

        for x in range(self.height):
            result.push([(self.matrix[x][y] * scl) for y in range(self.width)])
        return result

    def negate(self):
        return self.scalar(-1)

    __neg__ = negate

    @property
    def inverse(self):
        self.ensure_square()

        result = Matrix(self.height, self.width)
        rdet = 1 / self.det
        t = self.transpose()
        for x in range(self.height):
            for y in range(self.width):
                result.push(rdet * t.minor(x, y) * math.cos((x + y) * math.pi))
        return result

    inv = inverse

    def solve_system(self, vector):
        vector.ensure_vector()
        return self.inv * vector

    @property
    def trace(self):
        self.ensure_square()
        total = 0

        for x in range(self.height):
            for y in range(self.width):
                if x == y:
                    total += self.matrix[x][y]
        return total

    def multiply(self, other_matrix):
        self.ensure_full()
        other_matrix.ensure_full()
        if self.width != other_matrix.height:
            raise ValueError("Incompatible multiplication")

        product = create_array(self.height, other_matrix.width)

        for y in range(self.height):
            row = self.get_row(y)
            for x in range(other_matrix.width):
                product.push(row.dot(other_matrix.get_column(x)))

        return product

    __mul__ = multiply

    def divide(self, other_matrix):
        return self.multiply(other_matrix.inv)

    __div__ = divide

    def hadamard_product(self, other_matrix):
        self.ensure_equal_dims(other_matrix)

        result = create_array(self.height, self.width)
        for x in range(self.height):
            result.push([(self.matrix[x][y] * other_matrix.matrix[x][y]) for y in range(self.width)])
        return result

    hadp = hadamard_product

    def matrix_quotient(self, other_matrix):
        self.hadamard_product(other_matrix.element_raise_to(-1))
        return self.hadamard_product(other_matrix.element_raise_to(-1))

    def element_add(self, n):
        return self + Matrix.ones(self.height, self.width).scalar(n)

    def element_raise_to(self, n):
        result = create_array(self.height, self.width)
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] != 0:
                    result.push(self.matrix[x][y] ** n)
                else:
                    result.push(0)
        return result

    def raise_to(self, n):
        self.ensure_square()
        result = Matrix.identity(self.height)
        if n > 0:
            for x in range(n):
                result *= self
        elif n < 0:
            inverse = self.inv
            for x in range(-n):
                result *= inverse

        return result

    __pow__ = raise_to

    def kronecker_product(self, other_matrix):
        result = Matrix(self.height * other_matrix.height, self.width * other_matrix.width)

        for x in range(self.height):
            for y in range(self.width):
                for a in range(other_matrix.height):
                    for b in range(other_matrix.width):
                        result.set_item(self.matrix[x][y] * other_matrix.matrix[a][b], (x * self.height) + a, (y * self.width) + b)

        return result

    @property
    def is_skew_symmetric(self):
        if self.transpose().matrix == self.scalar(-1).matrix:
            return True
        else:
            return False

    @property
    def is_square(self):
        if self.width == self.height:
            return True
        else:
            return False

    @property
    def is_vector(self):
        if self.width == 1:
            return True
        else:
            return False

    @classmethod
    def rotation(cls, theta, degrees):
        if degrees:
            theta = math.radians(theta)
        result = cls(2, 2)
        result.push([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)])
        return result

    def allow_concatenation(self, other_matrix, dim):
        if not isinstance(other_matrix, Matrix):
            other_matrix = input_array(other_matrix)
        if eval("self." + dim) != eval("other_matrix." + dim):
            other_matrix = other_matrix.transpose()
        return other_matrix

    def horizontal_concatenate(self, other_matrix, dim="heights"):
        other_matrix = self.allow_concatenation(other_matrix, "height")
        self.ensure_equal_heights(other_matrix, dim)

        if self.has_values:
            result = create_array(self.height, self.width + other_matrix.width)
            for x in range(self.height):
                result.push([y for y in self.matrix[x] + other_matrix.matrix[x]])
        else:
            result = other_matrix
        return result

    hcon = horizontal_concatenate

    def vertical_concatenate(self, other_matrix):
        other_matrix = self.allow_concatenation(other_matrix, "width")
        return self.transpose().horizontal_concatenate(other_matrix.transpose(), "widths").transpose()

    vcon = vertical_concatenate

    def overwrite(self, other_matrix, row, col):
        if other_matrix.width > self.width or other_matrix.height > self.height:
            raise ValueError("Overwrite matrix is larger than the matrix to overwrite")
        result = create_array(self.height, self.width)
        result.matrix = self.matrix

        for x in range(self.height):
            for y in range(self.width):
                if other_matrix.width > x - row >= 0 and other_matrix.height > y - col >= 0:
                    result.set_item(other_matrix.matrix[x - row][y - col], x, y)
        return result

    def expand(self, n=3):
        if self.height != 2 or self.width != 2:
            raise ValueError("Matrix expansion is only defined for 2x2s")
        result = Matrix.identity(n)
        result.set_item(self.matrix[0][0], 0, 0)
        result.set_item(self.matrix[0][1], 0, -1)
        result.set_item(self.matrix[1][0], -1, 0)
        result.set_item(self.matrix[1][1], -1, -1)
        return result

    def is_greater_than(self, other_matrix):
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] < other_matrix.matrix[x][y]:
                    return False
        return True

    def is_less_than(self, other_matrix):
        return not self.is_greater_than(other_matrix)

    def is_equal_to(self, other_matrix):
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] != other_matrix.matrix[x][y]:
                    return False
        return True

    __eq__ = is_equal_to

    @property
    def sum_values(self):
        total = 0
        for x in range(self.height):
            for y in range(self.width):
                total += self.matrix[x][y]
        return total

    @classmethod
    def random_matrix(cls, height, width, value_range=[-20, 20]):
        result = cls(height, width)
        for x in range(height):
            for y in range(width):
                result.set_item(random.randint(*value_range), x, y)
        return result

    rand = random_matrix

    def round_matrix(self, decimal_places=0):
        result = create_array(self.height, self.width)

        for x in range(self.height):
            for y in range(self.width):
                result.push(round(self.matrix[x][y], decimal_places))

        return result

    @property
    def eigenvalues(self):
        self.ensure_square()
        if self.height == 2:
            return np.roots([1, -self.trace, self.det])
        elif self.height == 3:
            return np.roots([1, -self.trace, self.raise_to(2).trace - (self.trace() ** 2), -self.det])
        else:
            raise ValueError("Eigenvalues can currently only be found for 2x2s and 3x3s")

    eig = eigenvalues

    @property
    def eigenvectors(self):
        eigenvalues = self.eig
        result = Matrix(len(eigenvalues), len(eigenvalues))

        for i, eig in enumerate(eigenvalues):
            x = self - Matrix.identity.scalar(eig)
            if len(eigenvalues) == 2:
                eigenvector = x.get_col_as_vec(0)
            elif len(eigenvalues) == 3:
                eigenvector = x.get_col_as_vec(0).cross(x.get_col_as_vec(1))
            result.overwrite(eigenvector, 0, i)


class InputMatrix(Matrix):
    def __init__(self, *array):
        if len(array) > 1:
            array = [x for x in array]
        else:
            array = array[0]
        height = len(array)
        width = len(array[0])
        for x in range(height):
            if len(array[x]) != width:
                raise ValueError("Inconsistent matrix row lengths")

        super().__init__(height, width)
        for x in array:
            for y in x:
                self.push(y)


class Vector(Matrix):
    def __init__(self, n):
        super().__init__(n, 1)

    def get_vector_item(self, n):
        return self.vector(n)

    def magnitude(self, norm=2): # add property decorator?
        if norm == "inf":
            return max([abs(x) for x in self.vector])
        else:
            return self.element_raise_to(norm).sum_values ** (1 / norm)

    def normalize(self):
        norm_vec = Vector(self.height)
        for x in range(self.height):
            norm_vec.push(x / self.magnitude())
        return norm_vec

    @staticmethod
    def unit_vector(theta, degrees=False):
        if degrees:
            theta = math.radians(theta)
        return InputVector([math.cos(theta), math.sin(theta)])

    def dot_product(self, other_vector):
        other_vector.ensure_vector()
        total = 0
        for i in range(self.height):
            total += float(self.vector[i]) * float(other_vector.vector[i])
        return round(total, 4)

    dot = dot_product

    def outer_product(self, other_vector):
        other_vector.ensure_vector()
        return self * other_vector.transpose()

    outer = outer_product

    def get_angle(self, other_vector, deg=False):
        other_vector.ensure_vector()
        angle = math.acos(self.dot_product(other_vector) / (self.magnitude() * other_vector.magnitude()))
        if deg:
            return math.degrees(angle)
        else:
            return angle

    def scalar_projection(self, other_vector):
        other_vector.ensure_vector()
        return self.dot_product(other_vector) / (other_vector.magnitude())

    def vector_projection(self, other_vector):
        return other_vector.normalize().scalar(self.scalar_projection(other_vector))

    def cross_product(self, other_vector):
        other_vector.ensure_vector()
        if self.num_items() == 3 and other_vector.num_items() == 3:
            cross_product = Vector(3)
            for x in range(3):
                cross_product.push((self.vector[x] * other_vector.vector[(x + 1) % 3]) - (self.vector[(x + 1) % 3] * other_vector.vector[x]))

            return cross_product

        else:
            raise ValueError("The cross product is only defined for 3D vectors")

    cross = cross_product

    def diagonal(self):
        return Matrix.identity(self.height).hadamard_product(self.outer_product(Matrix.ones(self.height, 1)))

    def ensure_ndims(self, n):
        if self.height != n:
            raise ValueError("Must input a " + str(n) + "D vector")

    @staticmethod
    def get_basis(dim, n):
        result = Vector(dim)
        for x in range(dim):
            if x == n:
                result.push(1)
            else:
                result.push(0)
        return result

    def rotate_2d(self, theta, degrees=True):
        self.ensure_ndims(2)
        return Matrix.rotation(theta, degrees) * self

    def rotate_3d(self, rotation_vector, degrees=True):
        def x_rotation(vector, theta):
            return Matrix.identity(3).overwrite(Matrix.rotation(theta, degrees), 1, 1) * vector

        def y_rotation(vector, theta):
            return Matrix.rotation(theta, degrees).expand(3) * vector

        def z_rotation(vector, theta):
            return Matrix.identity(3).overwrite(Matrix.rotation(theta, degrees), 0, 0) * vector

        rotation_vector.ensure_vector()
        if self.height != 3 or rotation_vector.height != 3:
            raise ValueError("Must input 3D vectors")

        return x_rotation(y_rotation(z_rotation(self, rotation_vector.vector[2]), rotation_vector.vector[1]), rotation_vector.vector[0])

    def stack(self, num, horizontal=True):
        result = Vector(self.height)
        for _ in range(num):
            result = result.horizontal_concatenate(self)
        if horizontal:
            return result
        else:
            return result.transpose()

    def distance(self, other_vector, norm=2):
        other_vector.ensure_vector()
        return (other_vector - self).magnitude(norm)

    def getpoint(self, other_vector, proportion):
        other_vector.ensure_vector()
        return self + (other_vector - self).scalar(proportion)

    def midpoint(self, other_vector):
        return self.getpoint(other_vector, 0.5)

    def polar(self, degrees=False):
        self.ensure_ndims(2)
        x, y = self.vector
        theta = math.atan(y / x)
        if degrees:
            theta = math.degrees(theta)
        return InputVector([self.magnitude(), theta])

    def poltorect(self, degrees=False):
        self.ensure_ndims(2)
        r, theta = self.vector
        if degrees:
            theta = math.radians(theta)
        return InputVector([math.cos(theta), math.sin(theta)]).scalar(r)

    def spherical(self, degrees=False):
        self.ensure_ndims(3)
        x, y, z = self.vector
        phi, theta = math.atan(y / x), math.atan((((x ** 2) + (y ** 2)) ** 0.5) / z)
        if degrees:
            phi, theta = math.degrees(phi), math.degrees(theta)
        return InputVector([self.magnitude(), phi, theta])

    def sphtorect(self, complements=False, degrees=False):
        self.ensure_ndims(3)
        r, phi, theta = self.vector
        if degrees:
            phi = math.radians(phi)
            theta = math.radians(theta)
        if complements:
            phi = (math.pi / 2) - phi
            theta = (math.pi / 2) - theta
        return InputVector([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]).scalar(r)

    def cylindrical(self, degrees=False):
        x, y, z = self.vector
        self.ensure_ndims(3)
        phi = math.atan(y / x)
        if degrees:
            phi = math.degrees(phi)
        return InputVector([((x ** 2) + (y ** 2)) ** 0.5], phi, z)

    def cyltorect(self, degrees=False):
        self.ensure_ndims(3)
        rho, phi, z = self.vector
        if degrees:
            phi = math.radians(phi)
        return InputVector([rho * math.cos(phi), rho * math.sin(phi), z])

    def complex_multiply(self, other_vector):
        other_vector.ensure_vector()
        self.ensure_ndims(2)
        other_vector.ensure_ndims(2)
        theta1 = math.atan(self.vector[1] / self.vector[0])
        theta2 = math.atan(other_vector[1] / other_vector[0])
        return Vector.unit_vector(theta1 + theta2).scalar(self.magnitude() * other_vector.magnitude())


class InputVector(Vector):
    def __init__(self, array):
        super().__init__(len(array))
        self.push(array)


class Iteration:
    def __init__(self, state, transition):
        state.ensure_vector()
        self.state = state
        self.transition = transition

    def iterate(self, amount):
        for x in range(amount):
            self.state *= self.transition
        return self.state.matrix

    def iterate_until(self, threshold_vector):
        iterations = 0

        while self.state.is_greater_than(threshold_vector):
            self.state *= self.transition
            iterations += 1

        return iterations


def create_array(height, width):
    if width == 1:
        return Vector(height)
    elif width > 1:
        return Matrix(height, width)


def input_array(*array):
    if type(array[0]) == list:
        return InputMatrix(array)
    else:
        return InputVector(array)
