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
        s = lalib.Matrix(3, 4)
        self.assertTrue(t.ensure_equal_dims(s))
        self.assertFalse(s.has_values)


if __name__ == '__main__':
    unittest.main()
