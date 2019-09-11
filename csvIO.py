# coding:utf-8

import os
import re
from datetime import datetime

# /////////////////////////////////////////////////////////////////////////////
# [Csv-file reader/writer] ////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
class CsvIO:
    __header_skipped = False
    __list2D = []
    header = []

    def __init__(self, header_skipped=False):
        self.__header_skipped = header_skipped

    def read(self, file_name, mode='r'):
        file_name_del_spc = file_name.strip()
        if os.path.isfile(file_name_del_spc):
            with open(file_name_del_spc, mode, encoding="utf-8") as f:
                for line in f:
                    if not self.__header_skipped:
                        self.__header_skipped = True
                        self.header = line.rstrip().split(',')
                        continue
                    line = line.rstrip().split(',')
                    self.__list2D.append(line)
        else:
            print ("Error: No-file")
        return self.__list2D

    def get_column_data(self, header_name, err_msg=""):
        tmp = []
        count_header = len(self.header)
        if count_header == 0:
            return tmp
        values = list(range(0, count_header))
        dct = dict(zip(self.header, values))
        index = dct.get(header_name)
        for row in self.__list2D:
            if row[0] != err_msg:
                tmp.append(row[index])
        return tmp

# /////////////////////////////////////////////////////////////////////////////
# [Assertion] /////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
def AssertEqualValue(x, y):
    try:
        assert (x == y), ("expected:[{0}], actual:[{1}]".format(x,y))
    except AssertionError as err:
        print ("AssertionError :", err)

def AssertequalListSize(x, *ys):
    try:
        for y in ys:
            assert (len(x) == len(y)), ("expected:[{0}], actual:[{1}]".format(len(x),len(y)))
    except AssertionError as err:
        print ("AssertionError :", err)

# /////////////////////////////////////////////////////////////////////////////
# [Utils] /////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
def GetValueWithDefault(org_value, default_value = "-1"):
    return default_value if org_value == "" else org_value

def SortStrNums(values, is_AtoZ=True):
    num_expr = "[0-9]+"
    sign = 1
    if is_AtoZ != True:
        sign = -1
    return sorted(values, key=lambda x:sign*int((re.search(num_expr,x)).group(0)))

def ConvertToDate(value):
    date_regs = "\d{4}-\d{2}-\d{2}"
    date_expr = "%Y-%m-%d"
    return datetime.strptime((re.search(date_regs.value)).group(0),date_expr)

def SortStrDates(values, is_AtoZ=True):
    lst_values = [ConvertToDate(v) for v in values]
    sort_dates = sorted(lst_values)
    if is_AtoZ == True:
        return sort_dates
    else:
        return sort_dates[::-1]

def DateToStrs(dates):
    date_expr = "%Y-%m-%d"
    return [t.strftime(date_expr) for t in dates]
