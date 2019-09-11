""" This is utility functions program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True # dont make .pyc files

COLORS = {'navy': '000080', 'navyjp': '283455', 'orange': 'FF8000', 
          'white': 'FFFFFF', 'black': '000000'}

def get_os_name():
    """ display OS name """
    if os.name == 'nt':
        return 'on Windows'
    if os.name == 'posix':
        return 'on Mac or Linux'
    return None

def filter_odd_numbers(num):
    """ filter odd numbers """
    return num % 2 != 0

def filter_even_numbers(num):
    """ filter even numbers """
    return num % 2 == 0

if __name__ == '__main__':
    print(sys.prefix)
    print(sys.path)
    print(get_os_name())
    print(filter_odd_numbers(9))
    print(filter_odd_numbers(10))
    print(filter_even_numbers(99))
    print(filter_even_numbers(100))
