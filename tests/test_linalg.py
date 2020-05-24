import unittest
import linalgeuc.math.linear_algebra as lalib


class TestMatrix(unittest.TestCase):
    def test_init(self):
        t = lalib.Matrix(2, 3)
        self.assertEqual(t.height, 2)
        self.assertEqual(t.width, 3)

    def test_clear(self):
        t = lalib.Vector(3)
        t.push([3, 2, 1])
        self.assertEqual(len(t.matrix[0]), 1)
        self.assertEqual(len(t.vector), 3)

    def test_push(self):
        t = lalib.Vector(4)
        t.push([3, 2, 1, -1])
        self.assertEqual(t.matrix, [[3], [2], [1], [-1]])
        self.assertEqual(t.vector, [3, 2, 1, -1])
        t.clear()
        t.push(-10)
        self.assertIn(-10, t.matrix[0])

    def test_item(self):
        t = lalib.Matrix(2, 3)
        t.set_item(8, 1, 2)
        self.assertEqual(t.size, [2, 3])
        self.assertEqual(t.matrix[1][2], 8)
        t.change_item(1, 1, 2)
        self.assertEqual(t.matrix[1][2], 9)
        self.assertEqual(t.num_items, 6)
        self.assertEqual(t.get_item(1, 2), t.matrix[1][2])

    def test_get(self):
        t = lalib.Matrix(3, 2)
        t.push([5, 2, -12, 2, 7, -1])
        self.assertEqual(t.get_col(0).vector, [5, -12, 7])
        self.assertEqual(t.get_column_as_list(0), [5, -12, 7])
        self.assertEqual(t.get_row(1).vector, [-12, 2])
        self.assertEqual(t.get_row_as_list(2), [7, -1])

    def test_ensure(self):
        t = lalib.Matrix.random_matrix(3, 4)
        u = lalib.Matrix(3, 4)
        self.assertTrue(t.ensure_equal_dims(u))
        self.assertFalse(u.has_values)
        v = lalib.Vector(2)
        v.push([-5, 5])
        self.assertTrue(v.ensure_vector())

    def test_is_in(self):
        a = lalib.Matrix(2, 2)
        a.push([3, 1, 0, -2])
        self.assertTrue(a.is_in(1))

    def test_is_same(self):
        a = lalib.Matrix(2, 2)
        b = lalib.Matrix(2, 2)
        a.push([3, 1, -1, -2])
        b.push([3, 1, -1, -2])
        self.assertTrue(a.is_same(b))

    def test_input(self):
        t = lalib.InputMatrix([3, 4], [-1, 0])
        u = lalib.InputMatrix([[3, 4], [-1, 0]])
        self.assertTrue(t.is_same(u))

    def test_premade(self):
        i = lalib.Matrix.identity(4)
        self.assertEqual(i.get_row_as_list(2), [0, 0, 1, 0])
        o = lalib.Matrix.ones(4, 2)
        self.assertEqual(o.get_row_as_list(3), [1, 1])
        z = lalib.Matrix.zeros(2, 2)
        self.assertEqual(z.get_row_as_list(1), [0, 0])

    def test_transpose(self):
        t = lalib.Matrix(3, 2)
        t.push([1, 2, 3, 4, 5, 6])
        self.assertEqual(t.transpose().get_column_as_list(1), [3, 4])

    def test_add(self):
        t = lalib.Matrix(2, 4)
        t.push([-3, -2, -1, 0, 1, 2, 3, 4])
        u = lalib.Matrix(2, 4)
        u.push([1, 1, 1, 1, 0, 0, 0, -1])
        self.assertTrue((t + u).is_same(lalib.InputMatrix([-2, -1, 0, 1], [1, 2, 3, 3])))
        self.assertTrue((t - u).is_same(lalib.InputMatrix([-4, -3, -2, -1], [1, 2, 3, 5])))

    def test_submatrix(self):
        t = lalib.InputMatrix([5, 4, 3], [2, 1, 0])
        self.assertTrue(t.submatrix(0, 0, 1, 1).is_same(lalib.InputMatrix([5, 4], [2, 1])))
        self.assertTrue(t.submatrix(0, 2, 1, 2).is_same(lalib.InputVector([3, 0])))

    def test_scalar(self):
        t = lalib.InputMatrix([5, 10], [-2, -3])
        self.assertTrue(t.scalar(3).is_same(lalib.InputMatrix([15, 30], [-6, -9])))

    def test_inv(self):
        t = lalib.InputMatrix([3, -3, 4], [2, -3, 4], [0, -1, 1])
        self.assertEqual(1, t.det)
        self.assertTrue(t.inv.is_same(lalib.InputMatrix([1, -1, 0], [-2, 3, -4], [-2, 3, -3])))

    def test_solve_system(self):
        t = lalib.InputMatrix([1, 2, -1], [1, 3, 2], [2, 6, 1])
        u = lalib.InputVector([1, 7, 8])
        self.assertEqual([3, 0, 2], t.solve_system(u).vector)

    def test_trace(self):
        t = lalib.InputMatrix([3, 4], [-1, -2])
        self.assertEqual(t.trace, 1)

    def test_multiply(self):
        t = lalib.Matrix(2, 4)
        t.push([-3, -2, -1, 0, 1, 2, 3, 4])
        u = lalib.Matrix(4, 2)
        u.push([1, 1, 1, 1, 0, 0, 0, -1])
        self.assertTrue((t * u).is_equal_to(lalib.InputMatrix([-5, -5], [3, -1])))

    def test_round_matrix(self):
        t = lalib.InputMatrix([4 / 5, -3 / 2], [0.843, 12 / 7])
        self.assertTrue(t.round_matrix(2).is_equal_to(lalib.InputMatrix([0.8, -1.5], [0.84, 1.71])))

    def test_hadamard_product(self):
        t = lalib.InputMatrix([2, 3], [-1, 5], [9, 0])
        u = lalib.InputMatrix([2, -8], [-4, 6], [2, -1])
        self.assertTrue(t.hadp(u).is_equal_to(lalib.InputMatrix([4, -24], [4, 30], [18, 0])))
        self.assertTrue(t.matrix_quotient(u).round_matrix(1).is_equal_to(lalib.InputMatrix([1, -0.4], [0.2, 0.8], [4.5, 0])))

    def test_element_ops(self):
        t = lalib.InputMatrix([3, 3, 1], [-9, -3, 1])
        self.assertTrue(t.element_add(2).is_equal_to(lalib.InputMatrix([5, 5, 3], [-7, -1, 3])))
        self.assertTrue(t.element_raise_to(2).is_equal_to(lalib.InputMatrix([9, 9, 1], [81, 9, 1])))

    def test_raise_to(self):
        t = lalib.InputMatrix([3, 4], [-2, 1])
        self.assertTrue(t.raise_to(3).is_equal_to(lalib.InputMatrix([-29, 20], [-10, -39])))


if __name__ == '__main__':
    unittest.main()
