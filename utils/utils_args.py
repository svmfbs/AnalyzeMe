""" 
This is arguments program 
"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import inspect
import math
import os
import sys
import traceback
from argparse import ArgumentParser

sys.dont_write_bytecode = True # dont make .pyc files

class Aaa():
    def __init__(self):
        self.x = 1
        self.y = 2

    def show_x(self):
        return self.x

    def show_y(self):
        return self.y

    def show_getattr(self):
        a = 'show_x'
        b = 'show_y'
        print(getattr(self, a))
        print(getattr(self, a)())
        print(getattr(self, b)())


def callable_test():
    def func_test():
        print('function')

    def ClassTest():
        pass

    str_test = 'str'
    print(callable(sys))
    print(callable(func_test))
    print(callable(ClassTest))
    print(callable(str_test))


def argparser_test():
    desc_failed = f'{__file__} [Args] [Options]\nDetailed options -h or --help'
    # make parser with this description
    parser = ArgumentParser(description=desc_failed)
    # add arguments which need to specify
    parser.add_argument('-q', '--query', type=str, dest='query', required=True, help='Search word')
    parser.add_argument('-w', '--worker', type=int, dest='worker', default=1, help='Mutitask number')
    parser.add_argument('-b', '--bool', action='store_false', dest='bool')
    # parse specified arguments
    args = parser.parse_args()
    # allocate dest to variables
    (query, worker, bool) = (args.query, args.worker, args.bool)
    if worker > 100:
        parser.error('too many workers')

    print(f'args:= {args}')
    print(f'store_false:= {bool}')


def inspect_test(ins='isclass'):
    dict_ins = {'isclass': inspect.isclass, 'isfunction': inspect.isfunction, 'ismethod': inspect.ismethod}
    module = inspect # module-name
    for a in inspect.getmembers(module, dict_ins[ins]):
        print(a[0])


def sys_exc_info_test():
    try:
        1/0
    except Exception:
        (exc_type, exc_value, exc_traceback) = sys.exc_info()
        print('================================')
        print(type(exc_type))
        print(exc_type)
        print('================================')
        print(type(exc_value))
        print(exc_value)
        print('================================')
        print(type(exc_traceback))
        print(exc_traceback)
        print('================================')
        traceback.print_tb(exc_traceback)
        print('================================')
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        exc = ''.join('' + x for x in lines)
        # tb = traceback.extract_tb(exc_traceback)
        # formattedTb = traceback.format_list(tb)
        # exc = ''.join('' + x for x in formattedTb)
        print(exc)


if __name__ == "__main__":
    # callable_test()
    # argparser_test()
    print('# [2020/01/20] -------------------------')
    # A = Aaa()
    # A.show_getattr()
    # inspect_test()
    sys_exc_info_test()
