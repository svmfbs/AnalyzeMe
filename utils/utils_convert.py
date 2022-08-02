""" This is convert program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import datetime as dt
sys.dont_write_bytecode = True # dont make .pyc files

class ConvertUtils():
    @classmethod
    def str_to_datefmt(cls, date_str, extractfmt, outfmt='%Y-%m-%d'):
        return dt.datetime.strptime(date_str, extractfmt).strftime(outfmt)

    @classmethod
    def datestr_to_datetime(cls, date_str, outfmt='%Y-%m-%d'):
        return dt.datetime.strptime(date_str, outfmt)

    @classmethod
    def extract_number(cls, msg, pattern=r'\D'):
        return re.sub(pattern, '', msg, flags=re.ASCII).strip()

    @classmethod
    def to_float(cls, num_str, pattern=r'^\D'):
        return float(cls.extract_number(num_str, pattern))

    @classmethod
    def to_int(cls, num_str, pattern=r'^\D'):
        return int(cls.extract_number(num_str, pattern))
    
    @classmethod
    def to_all_extract_number(cls, lst1d, pattern=r'\D'):
        return [cls.extract_number(x, pattern) for x in lst1d]

    @classmethod
    def to_all_string(cls, lst1d):
        return list(map(str, lst1d))

    @classmethod
    def to_all_int(cls, lst1d):
        return list(map(int, lst1d))

    @classmethod
    def to_all_float(cls, lst1d):
        return list(map(float, lst1d))


if __name__ == '__main__':
    date_str='01-01-2020'
    datestr = ConvertUtils.str_to_datefmt(date_str, '%m-%d-%Y')
    datetime = ConvertUtils.datestr_to_datetime(datestr)
    print(f'Type:{type(datestr)}, value:{datestr}')
    print(f'Type:{type(datetime)}, value:{datetime}')

    num_float = '3.1415926'
    num_int = '3'
    num_float = ConvertUtils.to_float(num_float)
    num_int = ConvertUtils.to_int(num_int)
    print(f'Type:{type(num_float)}, value:{num_float}')
    print(f'Type:{type(num_int)}, value:{num_int}')

