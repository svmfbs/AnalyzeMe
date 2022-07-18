""" This is erea for EPower """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.config import DEFAULT_PYPIRC
import os
import sys
from typing import KeysView
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


import ep_jepx as ej

sys.dont_write_bytecode = True  # dont make .pyc files


def download(url):
    file_name = os.path.basename(url)
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(file_name, 'wb') as file:
            for chunk in res.iter_content(chunk_size=1024):
                file.write(chunk)


def round_average(data: list, dp=2) -> float:
    return round(np.mean(data), dp)


@dataclass
class PlotTrait:
    label: str
    marker: str 
    markersize: int
    markeredgecolor: str
    linestyle: str
    linewidth: int
    color: str


if __name__ == '__main__':
    print(os.getcwd())
    file_name = 'spot_2022.csv'
    url = os.path.join(r'http://www.jepx.org/market/excel/', file_name)
    has_file = False
    if has_file == False:
        download(url)
    print(f'download: {url}')

    file_name = '/Users/hide/sample/epower/spot_2022.csv'
    df = pd.read_csv(file_name, encoding='shift_jis')
    header = df.columns.values.tolist()

    syst_area_cols = ej.EPowerArea.get_system_area_column()
    area_syst_names = syst_area_cols.keys()
    # for x in syst_area_cols.keys():
    #     print(x)

    dict_slot = ej.EPowerTime.get_time_slot_zone()

    pdu = ej.PandaUtils(df)

    # extract 1-column data from pd
    dates_str1d = pdu.extract_column_pd(0)
    slots_str1d = pdu.extract_column_pd(1)
    prices_area_str1d = {k: pdu.extract_column_pd(v) for k, v in syst_area_cols.items()}

    # transform data type for extracted 1-column data
    dates = [ej.EPowerTime.to_date_YMD(x) for x in dates_str1d]
    slots = [int(x) for x in slots_str1d]
    prices_area = {k: [float(x) for x in v] for k, v in prices_area_str1d.items()}

    dt_prices_slots = dict()
    for name in area_syst_names:
        dt_prices_slots[name] = ej.get_dict_date_prices(slots, dates, dict_slot, prices_area[name])

    dt_prices_mean = dict()
    for k, v in dt_prices_slots['system'].items():
        prices = dict()
        for name in area_syst_names:
            prices[name] = round_average(dt_prices_slots[name][k])
        dt_prices_mean[k] = prices

    area_names = [k for k in area_syst_names]
    zspd_hkd_tky = dict()
    zspd_tky_ksi = dict()
    print(f'date, {area_names}')
    for k, v in dt_prices_mean.items():
        spd_hkd_tky = round(v['hokkaido'] - v['tokyo'], 2)
        spd_tky_ksi = round(v['tokyo'] - v['kansai'], 2)
        zspd_hkd_tky[k] = spd_hkd_tky
        zspd_tky_ksi[k] = spd_tky_ksi
        print(k, spd_hkd_tky, [v[name] for name in area_syst_names])

    ms = 4
    lw = 1
    dict_plot_trait = {
        'system': PlotTrait('system', None, None, None, 'dotted', lw, 'k'),
        'hokkaido': PlotTrait('hokkaido', '*', ms, 'blue', '-', lw, 'blue'),
        'tohoku': PlotTrait('tohoku', 's', ms, 'purple', ':', lw, 'purple'),
        'tokyo': PlotTrait('tokyo', 'o', ms, 'forestgreen', '-', lw, 'forestgreen'),
        'chubu': PlotTrait('chubu', 'v', ms, 'yellow', ':', lw, 'yellow'),
        'hokuriku': PlotTrait('hokuriku', '^', ms, 'gray', ':', lw, 'gray'),
        'kansai': PlotTrait('kansai', '<', ms, 'magenta', ':', lw, 'magenta'),
        'chugoku': PlotTrait('chugoku', '>', ms, 'cyan', ':', lw, 'cyan'),
        'shikoku': PlotTrait('shikoku', 'd', ms, 'orange', ':', lw, 'orange'),
        'kyushu': PlotTrait('kyushu', 'p', ms, 'red', ':', lw, 'red'),
        }
    xs = [t for t in dt_prices_mean.keys()]
    zs1 = [zspd_hkd_tky[t] for t in xs]
    zs2 = [zspd_tky_ksi[t] for t in xs]
    for name in area_names:
        ys = [v[name] for v in dt_prices_mean.values()]
        t = dict_plot_trait[name]
        plt.plot(xs, ys, label=t.label, marker=t.marker, markersize=t.markersize, markeredgecolor=t.markeredgecolor, linestyle=t.linestyle, linewidth=t.linewidth, color=t.color)
    plt.bar(xs, zs1, alpha=0.5, align='center', color="powderblue", edgecolor="navy", linewidth=1)
    plt.bar(xs, zs2, alpha=0.5, align='edge', color="thistle", edgecolor="navy", linewidth=1)
    plt.grid()
    plt.legend()
    xylabel_fontsize = 16
    plt.xlabel('Historical date', fontsize=xylabel_fontsize)
    plt.ylabel('Spot price', fontsize=xylabel_fontsize)
    plt.xticks(xs[::7], rotation=90)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15)
    plt.show()
