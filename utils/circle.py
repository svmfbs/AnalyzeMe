""" This is test program """
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import math
import sys

import numpy as np

sys.dont_write_bytecode = True # dont make .pyc files

def circle_make(center_x, center_y, radius):
    point_num = 100
    circle_x = []
    circle_y = []
    for i in range(point_num + 1):
        theta = i*2*math.pi / point_num
        circle_x.append(center_x + radius * math.cos(theta))
        circle_y.append(center_y + radius * math.sin(theta))
    return (circle_x, circle_y)


if __name__ == "__main__":
    x = 1.0
    y = 1.0
    radius = 1.0
    (xs, ys) = circle_make(x, y, radius)
    for x, y in zip(xs, ys):
        print(x, y)

