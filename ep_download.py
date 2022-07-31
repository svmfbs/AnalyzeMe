""" This is erea for EPower """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

import ep_jepx as ej

sys.dont_write_bytecode = True  # dont make .pyc files


def download(url: str) -> None:
    file_name = os.path.basename(url)
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(file_name, 'wb') as file:
            for chunk in res.iter_content(chunk_size=1024):
                file.write(chunk)


def round_average(data: list, dp=2) -> float:
    return round(np.mean(data), dp)


def name_of_week(dt: datetime, is_capital: bool=False) -> str:
    weeks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    if is_capital:
        weeks = [w.upper() for w in weeks]
    return weeks[dt.weekday()]


@dataclass
class PlotTrait:
    label: str
    marker: str 
    markersize: int
    markeredgecolor: str
    linestyle: str
    linewidth: int
    color: str


class EPSpotPrice(ej.EPowerArea, ej.EPowerTime):
    def __init__(self, file_name: str, has_file: bool) -> None:
        self.file_name = file_name
        self.has_file = has_file
        self.local_file = os.path.join(os.getcwd(), file_name)

    def download_from_url(self, url_path: str) -> None:
        self.url = os.path.join(url_path, self.file_name)
        if not self.has_file:
            download(self.url)
        print(f'download: {self.url}')

    def set_dataframe_csv(self, fmt_code='shift_jis'):
        self.dataframe = pd.read_csv(self.local_file, encoding=fmt_code)
        self.header = self.dataframe.columns.values.tolist()
        self.pdu = ej.PandaUtils(self.dataframe)

    def get_area_names_with_system(self) -> list:
        self.syst_area_cols = self.get_system_area_column()
        return self.syst_area_cols.keys()

    def __extract_key_area_value_date_price_pd(self) -> None:
        self._dates_str1d = self.pdu.extract_column_pd(0)
        self._slots_str1d = self.pdu.extract_column_pd(1)
        assert len(self._dates_str1d) == len(self._slots_str1d), 'len(dates_str1d) == len(slots_str1d) is required'
        self._prices_area_str1d = {area: self.pdu.extract_column_pd(nc) for area, nc in self.syst_area_cols.items()}

    def convert_type_of_key_area_value_date_price_pd(self) -> dict:
        """ {key: value}:= {area_name: [p1, p2, .., p48, p1, p2, .., p48, ...,p1, p2, .., p48]}
        """
        self.__extract_key_area_value_date_price_pd()
        self.dates = [self.to_date_YMD(x) for x in self._dates_str1d]
        self.slots = [int(x) for x in self._slots_str1d]
        area_prices = {k: [float(x) for x in v] for k, v in self._prices_area_str1d.items()}
        return area_prices

    def all_areas_date_prices_48(self, area_prices: dict) -> dict:
        """ {key: value}:= {area_name: {date: [p1, p2, ...,p48]}}
        """
        area_names = self.get_area_names_with_system()
        dict_slot = self.get_time_slot_zone()
        dt_prices_slots = dict()
        for area in area_names:
            prices48 = area_prices[area]
            dt_prices_slots[area] = ej.get_dict_date_prices(self.slots, self.dates, dict_slot, prices48)
        return dt_prices_slots

    def date_all_areas_mean_price(self, dt_prices_slots: dict) -> dict:
        """ {key: value}:= {datetime, {area_name: average_price}}
        """
        area_names = self.get_area_names_with_system()
        dt_prices_mean = dict()
        for t in dt_prices_slots['system']:
            dt_prices_mean[t] = {n: round_average(dt_prices_slots[n][t]) for n in area_names}
        return dt_prices_mean

    def set_area_spreads(self, dt_prices_mean: dict) -> None:
        """ calculate area spreads
        """
        area_names = self.get_area_names_with_system()
        self.zspd_hkd_tky = dict()
        self.zspd_tky_ksi = dict()
        for k, v in dt_prices_mean.items():
            self.zspd_hkd_tky[k] = round(v['hokkaido'] - v['tokyo'], 2)
            self.zspd_tky_ksi[k] = round(v['tokyo'] - v['kansai'], 2)

    def print_all_data(self, dt_prices_mean: dict) -> None:
        area_names = self.get_area_names_with_system()
        print(f'date, {area_names}')
        for k, v in dt_prices_mean.items():
            print(k, name_of_week(k), self.zspd_hkd_tky[k], self.zspd_tky_ksi[k], [v[name] for name in area_names])


def main():
    url_path = r'http://www.jepx.org/market/excel/'
    file_name = 'spot_2022.csv'
    has_file = False
    spot = EPSpotPrice(file_name, has_file)
    spot.download_from_url(url_path)
    spot.set_dataframe_csv()
    area_syst_names = spot.get_area_names_with_system()
    # for x in spot.syst_area_cols.keys():
    #     print(x)

    prices_area = spot.convert_type_of_key_area_value_date_price_pd()
    dt_prices_slots = spot.all_areas_date_prices_48(prices_area)
    dt_prices_mean = spot.date_all_areas_mean_price(dt_prices_slots)
    spot.set_area_spreads(dt_prices_mean)
    spot.print_all_data(dt_prices_mean)


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
    zs1 = [spot.zspd_hkd_tky[t] for t in xs]
    zs2 = [spot.zspd_tky_ksi[t] for t in xs]
    for name in area_syst_names:
        ys = [v[name] for v in dt_prices_mean.values()]
        t = dict_plot_trait[name]
        plt.plot(xs, ys, label=t.label, marker=t.marker, markersize=t.markersize, markeredgecolor=t.markeredgecolor, linestyle=t.linestyle, linewidth=t.linewidth, color=t.color)
    plt.bar(xs, zs1, alpha=0.5, align='center', color="powderblue", edgecolor="navy", linewidth=1, label='hokkaido - tokyo')
    plt.bar(xs, zs2, alpha=0.5, align='edge', color="thistle", edgecolor="navy", linewidth=1, label='tokyo - kansai')
    plt.grid()
    # plt.legend()
    plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1, fontsize=10)
    xylabel_fontsize = 16
    plt.xlabel('Historical date [grid end:=Friday]', fontsize=xylabel_fontsize)
    plt.ylabel('Spot price [Yen/kWh]', fontsize=xylabel_fontsize)
    plt.xticks(xs[::7], rotation=90)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15)
    plt.show()


def main2():
    x = np.linspace(0, 10, 1000)
    y1 = np.sin(x)
    y2 = np.cos(x) 
    c1, c2 = 'blue', 'green'
    l1, l2 = 'sin', 'cos'
    xl1, xl2 = 'x', 'x'
    yl1, yl2 = 'sin', 'cos'

    #グラフを表示する領域を，figオブジェクトとして作成。
    fig = plt.figure(figsize = (10,6), facecolor='lightblue')

    #グラフを描画するsubplot領域を作成。
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    #各subplot領域にデータを渡す
    ax1.plot(x, y1, color=c1, label=l1)
    ax2.plot(x, y2, color=c2, label=l2)

    #各subplotにxラベルを追加
    # ax1.set_xlabel(xl1)
    ax1.axes.xaxis.set_ticklabels([])
    ax2.set_xlabel(xl2)

    #各subplotにyラベルを追加
    ax1.set_ylabel(yl1)
    ax2.set_ylabel(yl2)

    # 凡例表示
    ax1.legend(loc = 'upper right') 
    ax2.legend(loc = 'upper right') 
    
    ax1.grid()
    ax2.grid()
    
    pos1 = ax2.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0 + 0.06,  pos1.width, pos1.height]
    ax2.set_position(pos2) # set a new position

    plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    main()
    # main2()