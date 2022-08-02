""" This is functions program for Text """
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import utils_convert as uc
sys.dont_write_bytecode = True # dont make .pyc files

def get_home_path():
    home_drive = os.getenv("HOMEDRIVE")
    home_path = os.getenv("HOMEPATH")
    if home_drive is not None and home_path is not None:
        return home_drive + home_path
    else:
        return None

def get_special_folder_path(special_folder):
    home_path = get_home_path()
    if home_path is not None:
        return os.path.join(home_path, special_folder)
    else:
        return None

def get_desktop_path():
    return get_special_folder_path("Desktop")

def get_mydocument_path():
    return get_special_folder_path("Documents")

def join_to_text(lst1d, endl='\n'):
    return endl.join(map(str, lst1d))

def print_list1d(lst1d):
    for row in lst1d:
        print(str(row))

def print_list2d_excel(lst2d):
    for row in lst2d:
        for cell in row:
            print(str(cell.value) + ',', end='')
        print()

def print_list2d_normal(lst2d):
    for row in lst2d:
        for cell in row:
            print(str(cell) + ',', end='')
        print()

def print_list2d_x2_excel(lst2d_1st, lst2d_2nd):
    for (row1st, row2nd) in zip(lst2d_1st, lst2d_2nd):
        for cell in row1st:
            print(str(cell.value) + ',', end='')
        for cell in row2nd:
            print(str(cell.value) + ',', end='')
        print()

def print_list2d_x2_normal(lst2d_1st, lst2d_2nd):
    for (row1st, row2nd) in zip(lst2d_1st, lst2d_2nd):
        for cell in row1st:
            print(str(cell) + ',', end='')
        for cell in row2nd:
            print(str(cell) + ',', end='')
        print()

def print_separate_line(char='-', repeat_n=40):
    print(str(char)*repeat_n)


class TextUtils():
    def __init__(self, file_in, file_out, encoding='utf-8'):
        self.file_path_read = file_in
        self.file_path_write = file_out
        self.encoding = encoding

    def getall_without_header(self, lst1d, has_header):
        if has_header and len(lst1d) > 1:
            lst1d.pop(0)
        return lst1d

    def read_header(self, has_header, delim=' '):
        with open(self.file_path_read, "r", encoding=self.encoding) as fr:
            alllines = fr.readlines()
            if has_header and len(alllines) > 1:
                header = alllines.pop(0)
                return header.strip().split(delim)
            else:
                return None

    def read_all(self, has_header, delim=' '):
        output = list()
        with open(self.file_path_read, "r", encoding=self.encoding) as fr:
            lines = self.getall_without_header(fr.readlines(), has_header)
            for row in lines:
                cols = row.strip().split(delim)
                values = uc.ConvertUtils.to_all_string(cols)
                output.append(values)
        return output

    def read_column(self, index, has_header):
        output = list()
        with open(self.file_path_read, "r", encoding=self.encoding) as fr:
            lines = self.getall_without_header(fr.readlines(), has_header)
            for row in lines:
                cols = row.strip().split()
                values = uc.ConvertUtils.to_all_string(cols)
                if index < len(values):
                    output.append(values[index])
                else:
                    print("# Error: wrong index")
                    sys.exit()
        return output

    def write_all(self, output):
        with open(self.file_path_write, mode="w", encoding=self.encoding) as fw:
            text = join_to_text(output, '\n')
            fw.write(text)


if __name__ == '__main__':
    print(get_home_path())
    print(get_desktop_path())
    print(get_mydocument_path())
    print_separate_line()

    lst = [1, 2, 3, 4, 5]
    print_list1d(lst)
    print(join_to_text(lst, ' '))
    print_separate_line()

    file_path_read = r"/Users/sample/test.log"
    file_path_write = r"/Users/sample/test2.txt"

    ikey = 1
    has_header = False
    fs = TextUtils(file_path_read, file_path_write)
    output = fs.read_column(ikey, has_header)
    fs.write_all(output)
