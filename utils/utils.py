""" This is utility functions program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime as dt
import os
import pathlib
import pickle
import re
import subprocess
import sys

import myConst as mc

sys.dont_write_bytecode = True # dont make .pyc files

COLORS = {'navy': '000080', 'navyjp': '283455', 'orange': 'FF8000', 'white': 'FFFFFF', 'black': '000000'}

def filter_odd_numbers(num):
    """ filter odd numbers """
    return num % 2 != 0

def filter_even_numbers(num):
    """ filter even numbers """
    return num % 2 == 0

def swap(a, b):
    return (b, a)

class Dimension0D():
    def __init__(self, name):
        self.name = name

class Dimension1D(Dimension0D):
    def __init__(self, name, row_from, row_to):
        super().__init__(name)
        self.row0 = row_from
        self.rowN = row_to

class Dimension2D(Dimension1D):
    def __init__(self, name, row_from, row_to, col_from, col_to):
        super().__init__(name, row_from, row_to)
        self.col0 = col_from
        self.colN = col_to

class Dimension3D(Dimension2D):
    def __init__(self, name, row_from, row_to, col_from, col_to, width_from, width_to):
        super().__init__(name, row_from, row_to, col_from, col_to)
        self.wdt0 = width_from
        self.wdtN = width_to

def sort_list2d_by_keyindex1_excel(tgt2d, chg2d, keyTgt, keyChg):
    src2d = list()
    for tgt in tgt2d:
        tc1 = str(tgt[keyTgt].value)
        for chg in chg2d:
            sc1 = str(chg[keyChg].value)
            if tc1 == sc1:
                src2d.append(chg)
                break
    return src2d

def sort_list2d_by_keyindex1_normal(tgt2d, chg2d, keyTgt, keyChg):
    src2d = list()
    for tgt in tgt2d:
        tc1 = str(tgt[keyTgt])
        for chg in chg2d:
            sc1 = str(chg[keyChg])
            if tc1 == sc1:
                src2d.append(chg)
                break
    return src2d

def sort_list2d_by_keyindex2_excel(tgt2d, chg2d, key1, key2):
    src2d = list()
    for tgt in tgt2d:
        tc1 = str(tgt[key1].value)
        tc2 = str(tgt[key2].value)
        for chg in chg2d:
            sc1 = str(chg[key1].value)
            sc2 = str(chg[key2].value)
            if tc1 == sc1 and tc2 == sc2:
                src2d.append(chg)
                break
    return src2d

def sort_list2d_by_keyindex2_normal(tgt2d, chg2d, tpl2d_keyTgt, tpl2d_keyChg):
    if len(tpl2d_keyTgt) != len(tpl2d_keyChg) or len(tpl2d_keyTgt) != 2:
        print('# Error: wrong key')
        sys.exit()
    src2d = list()
    for tgt in tgt2d:
        tc1 = str(tgt[tpl2d_keyTgt[0]])
        tc2 = str(tgt[tpl2d_keyTgt[1]])
        for chg in chg2d:
            sc1 = str(chg[tpl2d_keyChg[0]])
            sc2 = str(chg[tpl2d_keyChg[1]])
            if tc1 == sc1 and tc2 == sc2:
                src2d.append(chg)
                break
    return src2d


class CommandOS():
    @classmethod
    def open_folder(cls, dir_path):
        subprocess.Popen(['explorer', dir_path])

    @classmethod
    def open_excel(cls, excel_file):
        subprocess.Popen(['start', excel_file], shell=True)

    @classmethod
    def get_files(cls, dir_path, pattern, is_toponly=False):
        p = pathlib.Path(dir_path)
        if is_toponly:
            # TODO: check top-only
            return list(p.glob(pattern))
        else:
            subpattern = r'**/' + pattern
            return list(p.glob(subpattern))

    @classmethod
    def get_os_name(cls):
        """ display OS name """
        if os.name == 'nt':
            return 'on Windows'
        if os.name == 'posix':
            return 'on Mac or Linux'
        return None

    @classmethod
    def get_file_mod_datetime(cls, file_path):
        last_modified_time = os.path.getmtime(str(file_path))
        return dt.datetime.fromtimestamp(last_modified_time)

    @classmethod
    def type_condition(cls, val):
        if type(val) is str:
            print('type is str')
        elif type(val) is int:
            print('type is int')
        else:
            print('type is not str or int')

    @classmethod
    def instance_condition(cls, val):
        if isinstance(val, str):
            print('type is str')
        elif isinstance(val, int):
            print('type is int')
        else:
            print('type is not str or int')

    @classmethod
    def remove_linefeed_code(cls, org, is_without_LF=True, pattern='[\r\n]+$'):
        return re.sub(pattern, '', org) if is_without_LF else org

    @classmethod
    def change_charcode(cls, inputfile=None, outputfile=None, from_char='SHIFT_JIS', to_char='UTF-8'):
        env = os.environ.copy()
        print(env)
        # TODO
        pass

    @classmethod
    def get_env_variables(cls, delim=':'):
        ''' display environment variables '''
        paths = os.environ['PATH'].split(delim)
        return '\n'.join(paths)

    @classmethod
    def get_current_directory(cls):
        ''' find out current working directory '''
        return os.getcwd()

    @classmethod
    def get_all_files(cls, curr_dir):
        ''' display all files in current working directory '''
        return os.listdir(curr_dir)


def get_back_dates_list(str_yyyyMMdd, days_back=0, dateformat='%Y%m%d'):
    if not isinstance(days_back, int):
        raise ValueError(f'days_back must be an integer. [days_back={days_back}]')
    if days_back < 0:
        raise ValueError(f'days_back must be 0 and over. [days_back={days_back}]')
    back_dates = []
    # {days_back, days_back-1, ,,, 1, 0}
    for back_day in range(days_back, -1, -1):
        date = dt.datetime.strptime(str_yyyyMMdd, dateformat)
        calc_date = date + dt.timedelta(days=-back_day)
        str_calc_date = calc_date.strftime('%Y-%m-%d')
        back_dates.append(str_calc_date)
    return back_dates


if __name__ == '__main__':
    print(sys.prefix)
    print(sys.path)
    print(CommandOS.get_os_name())
    print(filter_odd_numbers(9))
    print(filter_odd_numbers(10))
    print(filter_even_numbers(99))
    print(filter_even_numbers(100))
    mc.PI = 3.1415926
    print(mc.PI)
    cwd = CommandOS.get_current_directory()
    print(f'getcwd: {cwd}')
    envpaths = CommandOS.get_env_variables()
    print(envpaths)
    print(CommandOS.get_all_files(cwd))

    PATH = r'/Users/sample'
    FILE = r'utils_text.py'
    file_name = os.path.join(PATH, FILE)
    print(CommandOS.get_file_mod_datetime(PATH))
    CommandOS.type_condition('str')
    CommandOS.type_condition(100)
    CommandOS.type_condition([0, 1, 2])
    CommandOS.instance_condition('str')
    CommandOS.instance_condition(100)
    CommandOS.instance_condition([0, 1, 2])

    target_date = '20200124'
    days_back = 4
    try:
        print(get_back_dates_list(target_date, days_back))
    except ValueError as e:
        print(e)
