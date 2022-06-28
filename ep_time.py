""" This is time for EPower """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime, date
from ep_utils import get_key_from_value, padding_zero


sys.dont_write_bytecode = True  # dont make .pyc files


def dict_time_slot_zone():
    return {
        1: '00:00-00:30',
        2: '00:30-01:00',
        3: '01:00-01:30',
        4: '01:30-02:00',
        5: '02:00-02:30',
        6: '02:30-03:00',
        7: '03:00-03:30',
        8: '03:30-04:00',
        9: '04:00-04:30',
        10: '04:30-05:00',
        11: '05:00-05:30',
        12: '05:30-06:00',
        13: '06:00-06:30',
        14: '06:30-07:00',
        15: '07:00-07:30',
        16: '07:30-08:00',
        17: '08:00-08:30',
        18: '08:30-09:00',
        19: '09:00-09:30',
        20: '09:30-10:00',
        21: '10:00-10:30',
        22: '10:30-11:00',
        23: '11:00-11:30',
        24: '11:30-12:00',
        25: '12:00-12:30',
        26: '12:30-13:00',
        27: '13:00-13:30',
        28: '13:30-14:00',
        29: '14:00-14:30',
        30: '14:30-15:00',
        31: '15:00-15:30',
        32: '15:30-16:00',
        33: '16:00-16:30',
        34: '16:30-17:00',
        35: '17:00-17:30',
        36: '17:30-18:00',
        37: '18:00-18:30',
        38: '18:30-19:00',
        39: '19:00-19:30',
        40: '19:30-20:00',
        41: '20:00-20:30',
        42: '20:30-21:00',
        43: '21:00-21:30',
        44: '21:30-22:00',
        45: '22:00-22:30',
        46: '22:30-23:00',
        47: '23:00-23:30',
        48: '23:30-24:00',
    }


def calc_cyclic_hour(str_time='00:00'):
    '''
    calculate cyclic hour: 25:00 -> 01:00
    '''
    hour = str_time[:2]
    hour_cyclic = str(int(hour) % 24)
    return padding_zero(hour_cyclic)


def get_next_half_hour(starttime):
    '''
    get next time plus half-hour: 08:00 -> 08:30, 24:00 -> 24:30
    '''
    minutes = starttime[-2:]
    hour = calc_cyclic_hour(starttime)
    if minutes == '00':
        endtime = hour + ':30'
    elif minutes == '30':
        hour1 = str(int(hour) + 1)
        endtime = padding_zero(hour1) + ':00'
    else:
        endtime = starttime
        print('# ERROR: minutes must be 00 or 30')
    return endtime


def get_prev_half_hour(endtime):
    '''
    get previous time minus half-hour: 08:30 -> 08:00, 24:00 -> 23:30
    '''
    minutes = endtime[-2:]
    hour = calc_cyclic_hour(endtime)
    if minutes == '00':
        if hour == '00':
            return '23:30'
        hour1 = str(int(hour) - 1)
        starttime = padding_zero(hour1) + ':30'
    elif minutes == '30':
        starttime = padding_zero(hour) + ':00'
    else:
        starttime = endtime
        print('# ERROR: minutes must be 00 or 30')
    return starttime


def get_index_zone_start_end(starttime, endtime):
    '''
    get time-slot index corresponding to starttime:endtime, 08:00, 20:00
    '''
    value_zone = starttime + '-' + endtime
    index_zone = get_key_from_value(dict_time_slot_zone(), value_zone)
    return index_zone


def get_index_zone_start(starttime='08:00'):
    '''
    get time-slot index corresponding to starttime, 08:00
    '''
    endtime = get_next_half_hour(starttime)
    return get_index_zone_start_end(starttime, endtime)


def get_index_zone_end(endtime='20:00'):
    '''
    get time-slot index corresponding to endtime, 20:00
    '''
    starttime = get_prev_half_hour(endtime)
    return get_index_zone_start_end(starttime, endtime)


def get_peaktime_zone(starttime='08:00', endtime='20:00'):
    '''
    get peaktime zone: {17: '08:00-08:30', ,,, 40: '19:30-20:00'}
    '''
    index_start = get_index_zone_start(starttime)
    index_end = get_index_zone_end(endtime)
    condition = lambda k: index_start <= k and k <= index_end
    return {k:v for k, v in dict_time_slot_zone().items() if condition(k)}


def to_date_YMD(str_date, fmt='%Y/%m/%d'):
    '''
    change string date into data type with yyyy/MM/dd
    '''
    tdatetime = datetime.strptime(str_date, fmt)
    return date(tdatetime.year, tdatetime.month, tdatetime.day)


if __name__ == '__main__':
    print(os.getcwd())
    # for k, v in dict_time_slot_zone().items():
    #     print(k, v)
    print(to_date_YMD('2022/03/4'))
    print(to_date_YMD('2022-03-5', '%Y-%m-%d'))
    print(to_date_YMD('03-6-2022', '%m-%d-%Y'))
    print(to_date_YMD('7-03-2022', '%d-%m-%Y'))
    print('# -------------------------- ')
    times = [
        '00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00', '04:30',
        '05:00', '05:30', '06:00', '06:30', '07:00', '07:30', '08:00', '08:30', '09:00', '09:30',
        '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30',
        '15:00', '15:30', '16:00', '16:30', '17:00', '17:30', '18:00', '18:30', '19:00', '19:30',
        '20:00', '20:30', '21:00', '21:30', '22:00', '22:30', '23:00', '23:30', '24:00', '24:30',
        '25:00', '25:30', '26:00', '26:30', '27:00', '27:30', '28:00', '28:30', '29:00', '29:30',
    ]
    for x in times:
        # print(x, get_next_half_hour(x))
        print(x, get_prev_half_hour(x))
    print('# -------------------------- ')
    print(get_index_zone_start('08:00'), get_index_zone_end('20:00'))
    print(get_index_zone_start('08:30'), get_index_zone_end('20:30'))
    for k, v in get_peaktime_zone().items():
        print(k, v)

