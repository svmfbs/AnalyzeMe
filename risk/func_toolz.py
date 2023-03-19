""" Functional Programing using Toolz
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import toolz as tz

sys.dont_write_bytecode = True # dont make .pyc files



if __name__ == '__main__':
    print('# Functional programing !')

    a = lambda x: x *2
    b = lambda x: [x]
    f = tz.compose(a, b)
    print(f(5))
    g = tz.pipe(5, a, b)
    print(g)
    h = tz.reduce(lambda a, b: a+b, range(10))
    print(h)

