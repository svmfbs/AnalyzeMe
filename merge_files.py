""" This is practice program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd

sys.dont_write_bytecode = True  # dont make .pyc files


def main():
    '''
    This is used for combining JEPS spot files
    '''
    files = sorted(glob.glob('./data/jepx/*.csv'))
    file_number = len(files)
    csv_list = []
    for file in files:
        csv_list.append(pd.read_csv(file, encoding='shift_jis', skiprows=[1]))
    merge_csv = pd.concat(csv_list)
    print(merge_csv)
    merge_csv.to_csv('spot_merged.csv', encoding='shift_jis', index=False)
    print(file_number, ' 個のCSVファイルを結合完了！！')


if __name__ == '__main__':
    print(os.getcwd())
    main()
