""" This is erea for EPower """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

sys.dont_write_bytecode = True  # dont make .pyc files


def get_area_names():
    return ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]


def get_area_column(init_col=6):
    '''
    name and column number of area prices in JEPX_spot.csv
    '''
    area = get_area_names()
    return dict(zip(area, [init_col + v[0] for v in enumerate(area)]))


def get_system_area_column(system_col=5, init_col=6):
    '''
    name and column number of system price and area prices in JEPX_spot.csv
    '''
    syst_cols = {"system": system_col}
    area_cols = get_area_column(init_col)
    return dict(area_cols, **syst_cols)


if __name__ == '__main__':
    print(os.getcwd())
    print(get_area_names())
    print(get_area_column())
    print(get_system_area_column())

