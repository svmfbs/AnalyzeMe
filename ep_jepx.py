""" This is practice program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import csv
import math
import random
import json
import logging
import pickle
import pprint
from datetime import datetime, date
from collections import defaultdict
from pandas.io.parsers import TextFileReader
import pytz
import copy
import numpy as np
import pandas as pd
import codecs
import io
import matplotlib.pyplot as plt


sys.dont_write_bytecode = True  # dont make .pyc files


class EPowerArea:
    def __init__(self) -> None:
        pass

    @classmethod
    def get_area_names(cls):
        return ["hokkaido", "tohoku", "tokyo", "chubu", "hokuriku", "kansai", "chugoku", "shikoku", "kyushu"]

    @classmethod
    def get_area_column(cls, init_col=6):
        '''
        name and column number of area prices in JEPX_spot.csv
        '''
        area = cls.get_area_names()
        return dict(zip(area, [init_col + v[0] for v in enumerate(area)]))

    @classmethod
    def get_system_area_column(cls, system_col=5, init_col=6):
        '''
        name and column number of system price and area prices in JEPX_spot.csv
        '''
        syst_cols = {"system": system_col}
        area_cols = cls.get_area_column(init_col)
        return dict(area_cols, **syst_cols)


class EPowerTime:
    def __init__(self) -> None:
        pass

    @classmethod
    def get_time_slot_zone(cls):
        return {
            1: '00:00−00:30', 2: '00:30−01:00',
            3: '01:00−01:30', 4: '01:30−02:00',
            5: '02:00−02:30', 6: '02:30−03:00',
            7: '03:00−03:30', 8: '03:30−04:00',
            9: '04:00−04:30', 10: '04:30−05:00',
            11: '05:00−05:30', 12: '05:30−06:00',
            13: '06:00−06:30', 14: '06:30−07:00',
            15: '07:00−07:30', 16: '07:30−08:00',
            17: '08:00−08:30', 18: '08:30−09:00',
            19: '09:00−09:30', 20: '09:30−10:00',
            21: '10:00−10:30', 22: '10:30−11:00',
            23: '11:00−11:30', 24: '11:30−12:00',
            25: '12:00−12:30', 26: '12:30−13:00',
            27: '13:00−13:30', 28: '13:30−14:00',
            29: '14:00−14:30', 30: '14:30−15:00',
            31: '15:00−15:30', 32: '15:30−16:00',
            33: '16:00−16:30', 34: '16:30−17:00',
            35: '17:00−17:30', 36: '17:30−18:00',
            37: '18:00−18:30', 38: '18:30−19:00',
            39: '19:00−19:30', 40: '19:30−20:00',
            41: '20:00−20:30', 42: '20:30−21:00',
            43: '21:00−21:30', 44: '21:30−22:00',
            45: '22:00−22:30', 46: '22:30−23:00',
            47: '23:00−23:30', 48: '23:30−24:00',
        }

    @classmethod
    def to_date_YMD(cls, str_date, fmt='%Y/%m/%d'):
        '''
        change string date into data type with yyyy/MM/dd
        '''
        tdatetime = datetime.strptime(str_date, fmt)
        return date(tdatetime.year, tdatetime.month, tdatetime.day)


class PandaUtils():
    def __init__(self, df: TextFileReader) -> None:
        self.df = df

    def extract_column_pd(self, ncol=0):
        '''
        extract one column from pandas data
        '''
        str2d = self.df.iloc[:, [ncol]].values
        return [x for row in str2d for x in row]


class JEPXSpot():
    def __init__(self) -> None:
        pass


def get_price_slot_line(irow, target_slot, dict_slot, list_price, last_slot=48):
    '''
    get price within 48 slots
    '''
    assert irow < len(list_price), '# Error: irow < len(list_price) is required'
    for jc in dict_slot:
        if target_slot == jc:
            return list_price[irow]
        else:
            pass


def get_dict_date_prices(slots, dates, dict_slot, prices, last_slot=48):
    '''
    arrage row-oredered prices in 48 slots into column-ordered prices, then make dictionary of key:date, value:prices
    '''
    assert len(slots) == len(dates), '# Error: len(slots) == len(dates) is required'
    list_price = list()
    dict_date_prices = dict()
    for ir, slot in enumerate(slots):
        date = dates[ir]
        if slot == 1:
            list_price.clear()
        price = get_price_slot_line(ir, slot, dict_slot, prices)
        list_price.append(price)
        if slot == last_slot:
            dict_date_prices[date] = copy.deepcopy(list_price)
    return dict_date_prices


def write_area_spot_price_by_row_date_col_slot(area_name, writer, slots, dates, dict_slot, prices_area):
    '''
    write [date, prices in 48 slots] with header and footer into writer-stream
    '''
    header = [area_name, *[str(x) for x in dict_slot]]
    writer.writerow(header)
    dict_date_prices = get_dict_date_prices(slots, dates, dict_slot, prices_area[area_name])
    for k, v in dict_date_prices.items():
        # oneline = str(k) + ',' + ','.join([str(x) for x in v])
        writer.writerow([k, *v])
    footer = ['' for x in header]
    writer.writerow(footer)


def calc_return_list(price_list):
    '''
    calculate daily return by (price.current - price.prev) / price.prev
    '''
    daily_return = list()
    price_prev = price_list[0]
    for i, p in enumerate(price_list):
        if i > 0:
            price_curr = p
            return_1day = (price_curr - price_prev) / price_prev
            daily_return.append(return_1day)
            price_prev = p
    return daily_return


def calc_mid_value_list(bins, nround=2):
    '''
    calculate mid value by (value.current - value.prev) / 2.0
    '''
    class_values = list()
    bin_prev = bins[0]
    for i, b in enumerate(bins):
        if i > 0:
            bin_curr = b
            class_value = (bin_curr + bin_prev) / 2.0
            class_values.append(round(class_value, nround))
            bin_prev = b
    return class_values


def seasonal_func(t, mu, s0, s1, s2, s3, s4):
    '''
    seasonal function: f(t)=s0+mu.t+s1.sin(2PI.t)+s2.cos(2PI.t)+s3.sin(4PI.t)+s4.sin(4PI.t)
    '''
    const_term = s0 + mu*t
    x2pi = 2.0*math.pi
    x4pi = 4.0*math.pi
    frqcy_term = s1 * math.sin(x2pi*t) + s2 * math.cos(x2pi*t) + s3 * math.sin(x4pi*t) + s4 * math.sin(x4pi*t)
    return const_term + frqcy_term


def main():
    file_name = '/Users/hide/sample/data/spot_merged.csv'
    df = pd.read_csv(file_name, encoding='shift_jis')
    header = df.columns.values.tolist()
    # print(df)
    # print(header)
    # print(df.describe()) # stats data
    # sys.exit()

    syst_area_cols = EPowerArea.get_system_area_column()
    # for k, v in syst_area_cols.items():
    #     print(k, v)
    # sys.exit()

    dict_slot = EPowerTime.get_time_slot_zone()
    # for k, v in dict_slot.items():
    #     print(k, v)
    # print(df.iloc[:, [0, 1, syst_area_cols["hokkaido"], syst_area_cols["tokyo"], syst_area_cols["system"]]])
    # sys.exit()

    pdu = PandaUtils(df)

    # extract 1-column data from pd
    dates_str1d = pdu.extract_column_pd(0)
    slots_str1d = pdu.extract_column_pd(1)
    prices_area_str1d = {k: pdu.extract_column_pd(v) for k, v in syst_area_cols.items()}

    # transform data type for extracted 1-column data
    dates = [EPowerTime.to_date_YMD(x) for x in dates_str1d]
    slots = [int(x) for x in slots_str1d]
    prices_area = {k: [float(x) for x in v] for k, v in prices_area_str1d.items()}
    # print(prices_area['system'])
    # print(prices_area['hokkaido'])
    # sys.exit()

    # [output]: prices by row.date, col.slot
    # with open('data/spot_merged_adj.csv', 'w') as fout:
    #     writer = csv.writer(fout)
    #     for area in syst_area_cols:
    #         write_area_spot_price_by_row_date_col_slot(area, writer, slots, dates, dict_slot, prices_area)
    # sys.exit()

    area_name = 'hokkaido'
    dict_date_prices = get_dict_date_prices(slots, dates, dict_slot, prices_area[area_name])
    # for k, v in dict_date_prices.items():
    #     print(k, v[0], v[1], v[2], v[3], v[4], v[5])
    nb_slot = 23
    price_all_dates = [v[nb_slot] for v in dict_date_prices.values()]
    daily_return = calc_return_list(price_all_dates)
    limit_value = 0.98
    daily_return_cut = [x for x in daily_return if -limit_value < x and x < limit_value]
    print('# mean val: ', np.mean(daily_return_cut))
    print('# variance: ', np.var(daily_return_cut))
    print('# std dev : ', np.std(daily_return_cut))
    for k, v in dict_date_prices.items():
        print(k, v[nb_slot])
    sys.exit()

    # [plot]
    # fig, ax = plt.subplots()
    # n, bins, patches = ax.hist(daily_return_cut, bins=100, range=(-1, 1))
    # class_values = calc_mid_value_list(bins)

    # ax.set_title('Distribution of daily return:' + area_name + ', #=' + str(nb_slot + 1))
    # ax.set_xlabel('daily return')
    # ax.set_ylabel('Count')
    # class_sample = class_values[::5]
    # label = [str(x) for x in class_sample]
    # ax.set_xticks(class_sample)
    # ax.set_xticklabels(label, rotation=90)
    # ax.grid()
    # # print(sum(n), len(daily_return))
    # # print(class_values)
    # plt.show()


def main_seasonal():
    mu = 1.99
    s0 = 0.14
    s1 = 0.02
    s2 = 0.06
    s3 = 0.12
    s4 = 0.08
    t0 = 0.0
    tn = 5.0
    # dt = 0.0027
    dt = 0.1
    n = int((tn - t0) / dt)
    ts = [t0 + i*dt for i in range(n)]
    fs = [seasonal_func(t, mu, s0, s1, s2, s3, s4) for t in ts]
    # [plot]
    # plt.plot(ts, fs, marker="o", color = "red", linestyle = "--")
    plt.plot(ts, fs, marker=".", color = "blue", linestyle = "dotted")
    plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    flag = 1
    if flag == 1:
        main()
    elif flag == 2:
        main_seasonal()
    else:
        print("# END")
