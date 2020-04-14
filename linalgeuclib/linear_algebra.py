import math


class Matrix:
    def __init__(self, height, width):  # rows, columns
        self.height = height
        self.width = width
        self.clear()

    def clear(self):
        self.matrix = [[None for y in range(self.width)] for x in range(self.height)]
        self.cursor = 0

    def push(self, item):
        def push_single(item):
            try:
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

    def size(self):
        return self.height, self.width

    def num_items(self):
        return self.height * self.width

    def get_item(self, row, col):
        return self.matrix[row - 1][col - 1]

    def get_column(self, x):
        return [row[x] for row in self.matrix]

    def get_row(self, y):
        return self.matrix[y]

    def ensure_equal_dims(self, other_matrix):
        self.ensure_full()
        other_matrix.ensure_full()
        if self.width != other_matrix.width or self.height != other_matrix.height:
            raise ValueError("Corresponding matrix dimensions must be equal")
        return True

    def ensure_square(self):
        self.ensure_full()
        if self.height != self.width:
            raise ValueError("Matrix must be square")
        return True

    def ensure_vector(self):
        self.ensure_full()
        if self.width != 1 and self.height != 1:
            raise ValueError("Must input vector")
        return True

    def ensure_full(self):
        for x in range(self.height):
            for y in range(self.width):
                if self.matrix[x][y] is None:
                    raise ValueError("At least some elements of the matrix are empty")
        return True

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
        result = Matrix(n, m)
        for x in range(n):
            result.push([1 for y in range(m)])
        return result

    @staticmethod
    def zeros(n, m):
        result = Matrix(n, m)
        for x in range(n):
            result.push([0 for y in range(m)])
        return result

    def transpose(self):
        result = Matrix(self.width, self.height)
        for x in range(self.height):
            result.push([self.matrix[x][y] for y in range(self.width)])
        return result

    def add(self, other_matrix):
        self.ensure_equal_dims(other_matrix)

        result = Matrix(self.height, self.width)
        for x in range(self.height):
            result.push([(self.matrix[x][y] + other_matrix.matrix[x][y]) for y in range(self.width)])
        return result

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
        result = Matrix(self.height, self.width)

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
        if self.width != other_matrix.height:
            raise ValueError("Incompatible multiplication")

        product = Matrix(self.height, other_matrix.width)

        def multiply_lines(row, column):
            if len(row) != len(column):
                raise ValueError("Multiplication impossible")
            sum = 0
            for i in range(len(row)):
                sum += float(row[i]) * float(column[i])
            return round(sum, 4)

        for y in range(self.height):
            row = self.get_row(y)
            for x in range(other_matrix.width):
                column = other_matrix.get_column(x)

                product.push(multiply_lines(row, column))

        return product

    def hadamard_product(self, other_matrix):
        self.ensure_equal_dims(other_matrix)

        result = Matrix(self.height, self.width)
        for x in range(self.height):
            result.push([(self.matrix[x][y] * other_matrix.matrix[x][y]) for y in range(self.width)])
        return result

    def kronecker_product(self, other_matrix):
        result = Matrix(self.height * other_matrix.height, self.width * other_matrix.width)
        print(self.height * other_matrix.height, self.width * other_matrix.width)

        for x in range(self.height):
            for y in range(self.width):
                for a in range(other_matrix.height):
                    for b in range(other_matrix.width):
                        result.matrix[(x * self.height) + a][(y * self.width) + b] = (self.matrix[x][y] * other_matrix.matrix[a][b])

        return result

    def is_skew_symmetric(self):
        if self.transpose().matrix == self.scalar(-1).matrix:
            return True
        else:
            return False


class InputMatrix(Matrix):
    def __init__(self, *array):
        height = len(array)
        width = len(array[0])
        super().__init__(height, width)
        for x in array:
            self.push(x)


class Vector(Matrix):
    def __init__(self, n, v_type=True):
        assert type(v_type) == bool
        if v_type:
            height = n
            width = 1
        else:
            height = 1
            width = n

        super().__init__(height, width)

    def magnitude(self):
        sum = 0
        for x in range(self.height):
            for y in range(self.width):
                sum += (self.matrix[x][y]) ** 2
        return math.sqrt(sum)

    def normalize(self):
        norm_vec = Vector(self.height)
        for x in range(self.height):
            norm_vec.push(x / self.magnitude)    
        return norm_vec

    def dot_product(self, other_vector):
        other_vector.ensure_vector()
        return other_vector.multiply(self.transpose()).matrix[0][0]

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

    def projection(self, other_vector):
        other_vector.ensure_vector()
        return other_vector.scalar((self.dot_product(other_vector) / (other_vector.magnitude() ** 2)))

    def cross_product(self, other_vector):
        other_vector.ensure_vector()
        if self.num_items() == 3 and other_vector.num_items() == 3:
            cross_product = Vector(3)
            cross_product.push((self.matrix[1][0] * other_vector.matrix[2][0]) - (self.matrix[2][0] * other_vector.matrix[1][0]))
            cross_product.push((self.matrix[2][0] * other_vector.matrix[0][0]) - (self.matrix[0][0] * other_vector.matrix[2][0]))
            cross_product.push((self.matrix[0][0] * other_vector.matrix[1][0]) - (self.matrix[1][0] * other_vector.matrix[0][0]))
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


class InputVector(Vector):
    def __init__(self, array, v_type=True):
        height = len(array)
        super().__init__(height, v_type)
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
