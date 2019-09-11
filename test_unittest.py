""" This is unittest program """
#!/Users/hide/sample/bin/python3
#-*-coding:utf-8-*-
import unittest
from unittest import TestCase
import sys
import numpy as np
sys.dont_write_bytecode = True # dont make .pyc files


def tashizan(a, b):
    """ calculate (a+b) """
    return a + b

def hikizan(a, b):
    """ calculate (a-b) """
    return a - b

class TestKeisan(unittest.TestCase):
    """ test class of tashizan """
    def test_tashizan(self):
        """ test method for tashizan """
        value1 = 2
        value2 = 6
        expected = 8
        actual = tashizan(value1, value2)
        self.assertEquals(expected, actual)

class TestKeisan2(unittest.TestCase):
    """ test keisan2 """
    def test_hikizan(self):
        """ test method for hikizan """
        value1 = 2
        value2 = 12
        expected = -10
        actual = hikizan(value1, value2)
        self.assertEqual(expected, actual)

class QuadraticEquation(object):
    """ Quadratic equation class """
    def __init__(self):
        self.a = 1.0
        self.b = 2.0
        self.c = 1.0

    def calc_root(self):
        """ calculate root (root1 < root2) """
        x1 = (-1.0*self.b - np.sqrt(self.b*self.b-4.0*self.a*self.c))//(2.0*self.a)
        x2 = (-1.0*self.b + np.sqrt(self.b*self.b-4.0*self.a*self.c))//(2.0*self.a)
        return x1, x2

    def calc_value(self, x):
        """ calculate polynomial value """
        return self.a * np.power(x, 2.0) + self.b * x + self.c

class TestQuadraticEquations(unittest.TestCase):
    """ test quadratic equations """
    def setUp(self):
        print("setup")
        self.eq = QuadraticEquation()

    def test_calc_root(self):
        """ test method of s """
        expected = (-1.0, -1.0)
        actual = self.eq.calc_root()

        self.assertEqual(expected, actual)
        self.assertEqual((0.0, 0.0), (self.eq.calc_value(actual[0]), self.eq.calc_value(actual[1])))

    def test_calc_value(self):
        """ test method of calc_value() """
        expected = (4.0, 9.0)
        actual = (self.eq.calc_value(1.0), self.eq.calc_value(2.0))
        self.assertEqual(expected, actual)

    def tearDown(self):
        print("tearDown")
        del self.eq

if __name__ == "__main__":
    unittest.main()
