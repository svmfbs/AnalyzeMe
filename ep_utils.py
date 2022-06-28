""" This is time for EPower """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys


sys.dont_write_bytecode = True  # dont make .pyc files


def padding_zero(num2):
    if len(num2) == 1:
        return '0' + num2
    return num2


def get_key_from_value(dict, val):
    keys = [k for k, v in dict.items() if v == val]
    if keys:
        return keys[0]
    return None


if __name__ == '__main__':
    print(os.getcwd())
    data = {1:'apple', 2:'orange', 3:'pine'}
    key = get_key_from_value(data, 'pine')
    print(key)
    num2 = '3'
    print(padding_zero(num2))
