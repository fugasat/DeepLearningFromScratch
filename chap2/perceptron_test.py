import unittest
import chap2.perceptron as perceptron


class TestStringMethods(unittest.TestCase):

    def test_NOT(self):
        self.assertEqual(perceptron.NOT(0), 1)
        self.assertEqual(perceptron.NOT(1), 0)

    def test_AND(self):
        self.assertEqual(perceptron.AND(0, 0), 0)
        self.assertEqual(perceptron.AND(1, 0), 0)
        self.assertEqual(perceptron.AND(0, 1), 0)
        self.assertEqual(perceptron.AND(1, 1), 1)

    def test_NAND(self):
        self.assertEqual(perceptron.NAND(0, 0), 1)
        self.assertEqual(perceptron.NAND(1, 0), 1)
        self.assertEqual(perceptron.NAND(0, 1), 1)
        self.assertEqual(perceptron.NAND(1, 1), 0)

    def test_OR(self):
        self.assertEqual(perceptron.OR(0, 0), 0)
        self.assertEqual(perceptron.OR(1, 0), 1)
        self.assertEqual(perceptron.OR(0, 1), 1)
        self.assertEqual(perceptron.OR(1, 1), 1)

    def test_XOR(self):
        self.assertEqual(perceptron.XOR(0, 0), 0)
        self.assertEqual(perceptron.XOR(1, 0), 1)
        self.assertEqual(perceptron.XOR(0, 1), 1)
        self.assertEqual(perceptron.XOR(1, 1), 0)

    def test_INC(self):
        self.assertEqual(perceptron.INC(0, 0, 0, 0), [0, 0, 0, 1])
        self.assertEqual(perceptron.INC(0, 0, 0, 1), [0, 0, 1, 0])
        self.assertEqual(perceptron.INC(0, 0, 1, 0), [0, 0, 1, 1])
        self.assertEqual(perceptron.INC(0, 0, 1, 1), [0, 1, 0, 0])
        self.assertEqual(perceptron.INC(0, 1, 0, 0), [0, 1, 0, 1])
        self.assertEqual(perceptron.INC(0, 1, 0, 1), [0, 1, 1, 0])
        self.assertEqual(perceptron.INC(0, 1, 1, 0), [0, 1, 1, 1])
        self.assertEqual(perceptron.INC(0, 1, 1, 1), [1, 0, 0, 0])


    def test_ADD(self):
        self.assertEqual(perceptron.ADD(0, 0, 0, 0,
                                        0, 0, 0, 0),
                                        [0, 0, 0, 0])
        self.assertEqual(perceptron.ADD(0, 0, 0, 1,
                                        0, 0, 0, 1),
                                        [0, 0, 1, 0])
        self.assertEqual(perceptron.ADD(0, 0, 1, 1,
                                        0, 0, 0, 1),
                                        [0, 1, 0, 0])
        self.assertEqual(perceptron.ADD(0, 1, 1, 1,
                                        0, 1, 1, 1),
                                        [1, 1, 1, 0])

if __name__ == '__main__':
    unittest.main()