""" 
This is shell command program
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import shlex
import subprocess

sys.dont_write_bytecode = True # dont make .pyc files


def test_command():
    subprocess.run(['ls', '-l', '-a']) # run: for python3.5 or later
    cmd = 'ls -l -a'
    tokens = shlex.split(cmd)
    subprocess.run(tokens)
    cmd = 'ls'
    resobj = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    resbyte = resobj.communicate()[0]
    process = resbyte.decode('utf-8')
    print(f'コマンドは\n{process}です')


def main(file):
    cmd = 'nkf -g'
    tokens = shlex.split(cmd)
    subprocess.run(tokens)


if __name__ == "__main__":
    print(__doc__)
    file_path = r'~/sample/data/nikkei_stock_average_daily_jp.csv'
    main(file_path)
