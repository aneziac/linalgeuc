import math


class Matrix:
    def __init__(self, height, width):  # rows, columns
        self.height = height
        self.width = width
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
                self.matrix[self.cursor // self.width][self.cursor % self.width] = item
                self.cursor += 1

            except IndexError:
                raise ValueError("Matrix full")

        if isinstance(item, list):
            for x in item:
                push_single(x)
        else:
            push_single(item)

    def print(self, round_vals=False):
        for x in range(self.height):
            for y in range(self.width):
                if round_vals:
                    print(round(self.matrix[x][y]), end=' ')
                else:
                    print(self.matrix[x][y], end=' ')
            print(" ")
        print(" ")

    def size(self):
        return self.height, self.width

    def set_item(self, item, row, col=0):
        if isinstance(self, Vector):
            self.vector[row] = item
        self.matrix[row][col] = item

    def change_item(self, delta, row, col=0):
        if isinstance(self, Vector):
            self.vector[row] += delta
        self.matrix[row][col] += delta

    def num_items(self):
        return self.height * self.width

    def get_item(self, row, col):
        return self.matrix[row][col]

    def get_column(self, x):
        return [row[x] for row in self.matrix]

    def get_col_as_vec(self, x):
        return InputVector(self.get_column(x))

    def get_row(self, y):
        return self.matrix[y]

    def get_row_as_vec(self, y):
        return InputVector(self.get_row(y))

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

    def has_values(self):
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] is not None:
                    return True
        return False

    @staticmethod
    def identity(n):
        result = Matrix(n, n)
        for x in range(n):
            for y in range(n):
                if x == y:
                    result.push(1)
                else:
                    result.push(0)
        return result

    @staticmethod
    def ones(n, m):
        result = create_array(n, m)
        for x in range(n):
            result.push([1 for y in range(m)])
        return result

    @staticmethod
    def zeros(n, m):
        result = create_array(n, m)
        for x in range(n):
            result.push([0 for y in range(m)])
        return result

    def transpose(self):
        result = create_array(self.width, self.height)
        for x in range(self.width):
            result.push([self.matrix[y][x] for y in range(self.height)])
        return result

    def add(self, other_matrix):
        self.ensure_equal_dims(other_matrix)

        result = create_array(self.height, self.width)
        for x in range(self.height):
            result.push([(self.matrix[x][y] + other_matrix.matrix[x][y]) for y in range(self.width)])
        return result

    def subtract(self, other_matrix):
        return self.add(other_matrix.scalar(-1))

    def minor(self, row, col):
        if self.height != self.width or self.height <= 2:
            raise ValueError("Invalid minor")

        result = Matrix(self.height - 1, self.width - 1)
        for x in range(self.height):
            for y in range(self.width):
                if x != row and y != col:
                    result.push(self.matrix[x][y])
        return result

    def det(self):
        self.ensure_square()

        if self.height == 2:
            return self.matrix[0][0] * self.matrix[1][1] - (self.matrix[0][1] * self.matrix[1][0])

        sum = 0
        for y in range(self.width):
            sum += self.matrix[0][y] * self.minor(0, y).det() * math.cos(y * math.pi)
        return sum

    def scalar(self, scl):
        result = create_array(self.height, self.width)

        for x in range(self.height):
            result.push([(self.matrix[x][y] * scl) for y in range(self.width)])
        return result

    def inv(self):
        self.ensure_square()

        result = Matrix(self.height, self.width)
        rdet = 1 / self.det()
        t = self.transpose()
        for x in range(self.height):
            for y in range(self.width):
                result.push(rdet * t.minor(x, y).det() * math.cos((x + y) * math.pi))
        return result

    def solve(self, vector):
        vector.ensure_vector()
        return self.inv().multiply(vector)

    def multiply(self, other_matrix):
        self.ensure_full()
        other_matrix.ensure_full()
        if self.width != other_matrix.height:
            raise ValueError("Incompatible multiplication")

        product = create_array(self.height, other_matrix.width)

        def multiply_lines(row, column):
            total = 0
            for i in range(len(row)):
                total += float(row[i]) * float(column[i])
            return round(total, 4)

        for y in range(self.height):
            row = self.get_row(y)
            for x in range(other_matrix.width):
                column = other_matrix.get_column(x)
                product.push(multiply_lines(row, column))

        return product

    __mul__ = multiply

    def divide(self, other_matrix):
        return self.multiply(other_matrix.inv())

    def hadamard_product(self, other_matrix):
        self.ensure_equal_dims(other_matrix)

        result = create_array(self.height, self.width)
        for x in range(self.height):
            result.push([(self.matrix[x][y] * other_matrix.matrix[x][y]) for y in range(self.width)])
        return result

    def matrix_quotient(self, other_matrix):
        return self.hadamard_product(other_matrix.element_raise_to(-1))

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
        result = Matrix(self.height, self.width)
        for x in range(n):
            result.multiply(self)
        return result

    def kronecker_product(self, other_matrix):
        result = Matrix(self.height * other_matrix.height, self.width * other_matrix.width)
        print(self.height * other_matrix.height, self.width * other_matrix.width)

        for x in range(self.height):
            for y in range(self.width):
                for a in range(other_matrix.height):
                    for b in range(other_matrix.width):
                        result.set_item(self.matrix[x][y] * other_matrix.matrix[a][b], (x * self.height) + a, (y * self.width) + b)

        return result

    def is_skew_symmetric(self):
        if self.transpose().matrix == self.scalar(-1).matrix:
            return True
        else:
            return False

    @staticmethod
    def rotation(theta, degrees):
        if degrees:
            theta = math.radians(theta)
        result = Matrix(2, 2)
        result.push([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)])
        return result

    def horizontal_concatenate(self, other_matrix, dim="heights"):
        self.ensure_equal_heights(other_matrix, dim)
        if self.has_values():
            result = create_array(self.height, self.width + other_matrix.width)
            for x in range(self.height):
                result.push([y for y in self.matrix[x] + other_matrix.matrix[x]])
        else:
            result = other_matrix
        return result

    def vertical_concatenate(self, other_matrix):
        return self.transpose().horizontal_concatenate(other_matrix.transpose(), "widths").transpose()

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


class InputMatrix(Matrix):
    def __init__(self, *array):
        if len(array) == 1:
            array = array[0]
        height = len(array)
        width = len(array[0])
        for x in range(height):
            if len(array[x]) != width:
                raise ValueError("Inconsistent matrix row lengths")

        super().__init__(height, width)
        for x in array:
            self.push(x)


class Vector(Matrix):
    def __init__(self, n):
        super().__init__(n, 1)

    def get_vector_item(self, n):
        return self.vector(n)

    def magnitude(self):
        sum = 0
        for x in range(self.height):
            sum += (self.vector[x]) ** 2
        return math.sqrt(sum)

    def normalize(self):
        norm_vec = Vector(self.height)
        for x in range(self.height):
            norm_vec.push(x / self.magnitude)
        return norm_vec

    def dot_product(self, other_vector):
        other_vector.ensure_vector()
        return other_vector.multiply(self.transpose()).vector[0]

    dot = dot_product

    def outer_product(self, other_vector):
        other_vector.ensure_vector()
        return self.multiply(other_vector.transpose())

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
            def cross(first, second):
                cross_product.push((self.vector[first] * other_vector.vector[second]) - (self.vector[second] * other_vector.vector[first]))

            cross_product = Vector(3)
            cross(1, 2)
            cross(2, 0)
            cross(0, 1)
            return cross_product

        else:
            raise ValueError("The cross product is only defined for 3D vectors")

    def diagonal(self):
        return Matrix.identity(self.height).hadamard_product(self.outer_product(Matrix.ones(self.height, 1)))

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
        if self.height != 2:
            raise ValueError("Must input a 2D vector")
        return Matrix.rotation(theta, degrees).multiply(self)

    def rotate_3d(self, rotation_vector, degrees=True):
        def x_rotation(vector, theta):
            return Matrix.identity(3).overwrite(Matrix.rotation(theta, degrees), 1, 1).multiply(vector)

        def y_rotation(vector, theta):
            return Matrix.rotation(theta, degrees).expand(3).multiply(vector)

        def z_rotation(vector, theta):
            return Matrix.identity(3).overwrite(Matrix.rotation(theta, degrees), 0, 0).multiply(vector)

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
            self.state = self.transition.multiply(self.state)
        return self.state.matrix

    def iterate_until(self, threshold_vector):
        iterations = 0

        while self.state.is_greater_than(threshold_vector):
            self.state = self.transition.multiply(self.state)
            iterations += 1

        return iterations


def create_array(height, width):
    if width == 1:
        return Vector(height)
    elif width >= 1:
        return Matrix(height, width)