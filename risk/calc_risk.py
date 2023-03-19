""" This is VaR calculation
Ref. Anybody can do Value at Risk:
A Teaching Study using Parametric Computation and Monte Carlo Simulation, (2012)
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from datetime import datetime
from math import sqrt
from statistics import mean, stdev
from scipy.stats import norm, pearsonr
import numpy as np
from numpy import random as nr

import matplotlib
matplotlib.rcParams['backend'] = 'QtAgg'
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True # dont make .pyc files

nr.seed(250)


def read_file(file_name: str, encode: str) -> list[list[str]]:
    """ read csv file written with ASCII code
    """
    with open(file_name, 'r', encoding=encode) as f:
        kw_list = f.read().split('\n')
    lst2d = []
    for r in kw_list:
        cols = [c.strip(',') for c in r.split(' ') if len(c) > 0]
        if len(cols) > 0:
            lst2d.append(cols)
    return lst2d


def read_quote_csv(file_name: str, encode: str, remove_top: int=0) -> list[list[str]]:
    """ read csv file written shift-jis, removeing n lines from top
    """
    with open(file_name, 'r', encoding=encode) as f:
        kw_list = f.read().split('\n')
    lst2d = []
    for r in kw_list[remove_top:]:
        cols = [c.strip(' ') for c in r.split(',')]
        if len(cols) > 0:
            lst2d.append(cols)
    return lst2d


def read_fx_data(file_name: str, ccy: str) -> dict[datetime, float]:
    """ get historical fx prices
    """
    assert len(ccy) == 3, 'len(ccy) == 3 is required'
    lst2d = read_quote_csv(file_name, 'shift_jis', 2)
    header = lst2d[0]
    ccys = ['USD', 'GBP', 'EUR', 'CAD', 'CHF', 'SEK', 'DKK', 'NOK', 'AUD', 'NZD', 'ZAR', 'HKD', 'SGD']
    ncol_ccys = {c: header.index(c) for c in ccys}
    nc = ncol_ccys[ccy]
    dt_fx = {}
    for r in lst2d[1:]:
        if len(r[0]) > 0:
            dt = datetime.strptime(r[0], '%Y/%m/%d')
            # prices = [float(r[v]) for k, v in ncol_ccys.items()]
            dt_fx[dt] = float(r[nc])
    return dt_fx


def convert_str_to_float(str2d: list[list[str]]) -> list[list[float]]:
    """ convert str to float in list2d
    """
    flt2d = []
    for r in str2d:
        flt2d.append([float(c) for c in r if len(c) > 0])
    return flt2d


def calc_return(price_curr: float, price_prev: float) -> float:
    """ calculate return
    """
    assert price_prev > 0.0, 'prev > 0.0 is required'
    rtn = (price_curr - price_prev) / price_prev
    return rtn


def calc_all_return(prices: list[float]) -> list[float]:
    """ calculate all return without 1st item
    """
    all_return = []
    for i, s in enumerate(prices):
        if i > 0:
            pprev = prices[i-1]
            pcurr = prices[i]
            retrn = calc_return(pcurr, pprev)
            all_return.append(retrn)
    return all_return


def calc_correlation_matrix(all_returns: dict[int, list[float]]) -> np.array:
    """ calculate correlation matrix
    eg. [ [1, rho12, rho13], [rho21, 1, rho23], [rho31, rho32, 1] ]
    """
    nsmin = min(all_returns.keys())
    nsmax = max(all_returns.keys())
    assert nsmin == 1, 'nsmin == 1 is required'
    mat_corr = np.identity(nsmax)
    for ir in range(nsmin, nsmax+1):
        for jc in range(ir+1, nsmax+1):
            mat_corr[ir-1][jc-1] = pearsonr(all_returns[ir], all_returns[jc])[0]
            mat_corr[jc-1][ir-1] = mat_corr[ir-1][jc-1]
    return mat_corr


def VaR_single_asset(stock: list[float], amount: float, clevel: float, is_show: bool) -> tuple[float, float]:
    """ calculate Parametric VaR for single asset
    """
    print('# calculate VaR for single asset')
    price_last = stock[-1]
    all_rtn = calc_all_return(stock)
    rtn_mean = mean(all_rtn)
    rtn_stdev = stdev(all_rtn) # sigma, not sigma^2
    var_rtn = norm.ppf(1.0-clevel, rtn_mean, rtn_stdev)
    var_val = abs(var_rtn*amount)
    print(f'stock price      := {price_last}')
    print(f'mean of return   := {rtn_mean:.6f}')
    print(f'stdev of return  := {rtn_stdev:.6f}')
    print(f'confidence level := {clevel}')
    print(f'position amount  := {amount}')
    print(f'VaR(CL) return   := {var_rtn:.6f}')
    print(f'VaR(CL) value    := {var_val:.2f}')
    assert 38375.31 < var_val < 38375.33, '38375.31 < var_val < 38375.33 is required'
    # nmax = 2000
    # x_rvs = norm.rvs(size=nmax)
    if is_show:
        nobs = len(all_rtn)
        min_return = min(all_rtn)
        max_return = max(all_rtn)
        xs = np.linspace(min_return, max_return, nobs)
        ys = [norm.pdf(x, rtn_mean, rtn_stdev) for x in xs]
        zh = norm.pdf(var_rtn, rtn_mean, rtn_stdev)
        plt.grid(True)
        plt.plot(xs, ys)
        plt.vlines(var_rtn, 0, zh, color='r')
        plt.show()
    return (var_rtn, var_val)


def VaR_multiple_asset(stocks: dict[int, list[float]], amounts: list[float], clevel: float, is_show: bool) -> tuple[float, float]:
    """ calculate Parametric VaR for multiple assets
    """
    print('# calculate VaR for multiple asset portfolio')
    for k in stocks:
        assert len(stocks[1]) == len(stocks[k]), f'len(stocks[1]) == len(stocks[k]) is required'
    port_init_value = sum(amounts)
    port_weight = np.array([amt/port_init_value for amt in amounts])
    port_prices = np.array([stocks[k][-1] for k in stocks])
    all_returns = {k: calc_all_return(stocks[k]) for k in stocks}
    port_rtn_mean = np.array([mean(all_returns[k]) for k in all_returns])
    port_rtn_stdev = np.array([stdev(all_returns[k]) for k in all_returns])
    port_weight_stdev = np.array([x*y for x, y in zip(port_weight, port_rtn_stdev)])
    port_mean = port_weight.dot(port_rtn_mean)
    corr_mat = calc_correlation_matrix(all_returns)
    port_variance = port_weight_stdev.dot(corr_mat.dot(port_weight_stdev))
    port_stdev = sqrt(port_variance)
    var_rtn = norm.ppf(1.0-clevel, port_mean, port_stdev)
    var_val = abs(var_rtn*port_init_value)
    print(f'stock prices         := {port_prices}')
    print(f'mean of returns      := {[round(r,6) for r in port_rtn_mean]}')
    print(f'stdev of returns     := {[round(s,6) for s in port_rtn_stdev]}')
    print(f'mean of port return  := {port_mean:.6f}')
    print(f'correlation          := {corr_mat[0][1]:.6f}')
    print(f'stdev of port return := {port_stdev:.6f}')
    print(f'port initial amount  := {port_init_value}')
    print(f'port VaR(CL) return  := {var_rtn:.6f}')
    print(f'port VaR(CL) value   := {var_val:.2f}')
    assert 61949.54 < var_val < 61949.56, '61949.54 < var_val < 61949.56 is required'
    if is_show:
        nobs = len(all_returns[1])
        min_return = min(all_returns[1])
        max_return = max(all_returns[1])
        xs = np.linspace(min_return, max_return, nobs)
        ys = [norm.pdf(x, port_rtn_mean, port_rtn_stdev) for x in xs]
        zh = norm.pdf(var_rtn, port_rtn_mean, port_rtn_stdev)
        plt.grid(True)
        plt.plot(xs, ys)
        # plt.vlines(var_rtn, 0, zh, color='r')
        plt.show()
    return (var_rtn, var_val)


def plot_gaussian_density(nmc: int, mu: float, sigma: float, is_show: bool) -> None:
    """ plot normal random numbers histgram and Gaussian probability density
    """
    rand_gauss = np.random.default_rng().normal(mu, sigma, nmc)
    # rand_gauss = np.random.normal(mu, sigma, nmc)
    if is_show:
        count, bins, ignored = plt.hist(rand_gauss, 100, density=True)
        plt.plot(bins, norm.pdf(bins, mu, sigma),linewidth=2, color='r')
        plt.grid(True)
        plt.show()


def find_kth_value(vals: list[float], kth: int, from_small: bool=True) -> float:
    """ find k-th value in value sequence from small or large value
    """
    assert kth >= 0, 'kth >= 0 is required'
    assert len(vals) > kth, 'len(vals) > kth is required'
    vals_orderd = sorted(vals)
    if from_small:
        return vals_orderd[kth]
    else:
        vals_rev = vals_orderd[::-1]
        return vals_rev[kth]


def geometric_brownian_process(k1Y: float, sigma1Y: float, deltaT: float, w: float) -> float:
    """ S(t+dt) = S(t)exp(k.dt + sigma.sqrt(t).w)
    R(t+dt) = ln(S(t+dt)/S(t)) = k.dt + sigma.sqrt(t).w
    """
    delta_return = k1Y * deltaT + sigma1Y * sqrt(deltaT) * w
    return delta_return


def MCVaR_single_asset(nmc: int, stock: list[float], amount: float, clevel: float, days_trading: float, is_show: bool) -> tuple[float, float]:
    """ calculate Monte Carlo VaR for single asset
    """
    # print('# calculate Monte Carlo VaR for single asset')
    all_rtn = calc_all_return(stock)
    rtn_mean = mean(all_rtn)
    rtn_stdev = stdev(all_rtn) # sigma, not sigma^2
    deltaT = 1.0 / days_trading
    rtn_mean_1Y = rtn_mean * days_trading
    rtn_stdev_1Y = rtn_stdev * sqrt(days_trading)
    rtn_expected = rtn_mean_1Y - 0.5*rtn_stdev_1Y*rtn_stdev_1Y
    k1Y = rtn_expected
    sigma1Y = rtn_stdev_1Y
    rand_gauss = np.random.default_rng().normal(0.0, 1.0, nmc)
    rtn_sim = [geometric_brownian_process(k1Y, sigma1Y, deltaT, w) for w in rand_gauss]
    nobs_bottom = int((1.0 - clevel)*nmc) - 1
    val_kth = find_kth_value(rtn_sim, nobs_bottom)
    var_val_sim = abs(amount * val_kth)
    # print(f'VaR(CL) val sim  := {round(var_val_sim, 2)}')
    if is_show:
        count, bins, ignored = plt.hist(rtn_sim, 80, density=True)
        plt.show()
    return (val_kth, var_val_sim)


def MCVaR_multiple_asset(nmc: int, stocks: dict[int, list[float]], amounts: list[float], clevel: float, days_trading: float, is_show: bool) -> tuple[float, float]:
    """ calculate Monte Carlo VaR for multiple asset
    """
    # print('# calculate Monte Carlo VaR for multiple asset')
    port_init_value = sum(amounts)
    port_weight = np.array([amt/port_init_value for amt in amounts])
    port_prices = np.array([stocks[k][-1] for k in stocks])
    all_returns = {k: calc_all_return(stocks[k]) for k in stocks}
    port_rtn_mean = np.array([mean(all_returns[k]) for k in all_returns])
    port_rtn_stdev = np.array([stdev(all_returns[k]) for k in all_returns])
    corr_mat = calc_correlation_matrix(all_returns)
    deltaT = 1.0 / days_trading
    rtn_mean_1Y = [days_trading * rtn for rtn in port_rtn_mean]
    port_mean_1Y = port_weight.dot(rtn_mean_1Y)
    rtn_stdev_1Y = [sig * sqrt(days_trading) for sig in port_rtn_stdev]
    port_weight_stdev_1Y = np.array([w*v for w, v in zip(port_weight, rtn_stdev_1Y)])
    port_variance_1Y = port_weight_stdev_1Y.dot(corr_mat.dot(port_weight_stdev_1Y))
    port_stdev_1Y = sqrt(port_variance_1Y)
    rtn_expected = port_mean_1Y - 0.5*port_variance_1Y
    #
    k1Y = rtn_expected
    sigma1Y = port_stdev_1Y
    rand_gauss = np.random.default_rng().normal(0.0, 1.0, nmc)
    rtn_sim = [geometric_brownian_process(k1Y, sigma1Y, deltaT, w) for w in rand_gauss]
    nobs_bottom = int((1.0 - clevel)*nmc) - 1
    val_kth = find_kth_value(rtn_sim, nobs_bottom)
    var_val_sim = abs(port_init_value * val_kth)
    # print(f'port VaR(CL) val sim := {var_val_sim:.2f}')
    if is_show:
        count, bins, ignored = plt.hist(rtn_sim, 80, density=True)
        plt.show()
    return (val_kth, var_val_sim)


def check_VaR_accuracy(is_single: bool, nmc_set, nmax, stock, amount, clevel, days_trading, is_show) -> dict[int, list[float]]:
    """
    """
    if is_single:
        var_rtn1, var_val1 = VaR_single_asset(stock, amount, clevel, is_show)
    else:
        var_rtn1, var_val1 = VaR_multiple_asset(stock, amount, clevel, is_show)
    dict_diff = {}
    for nmc in nmc_set:
        diffs = []
        for m in range(nmax):
            if is_single:
                var_rtn_sim, var_val_sim = MCVaR_single_asset(nmc, stock, amount, clevel, days_trading, is_show)
            else:
                var_rtn_sim, var_val_sim = MCVaR_multiple_asset(nmc, stock, amount, clevel, days_trading, is_show)
            diff_var = abs(var_val1 - var_val_sim)
            diffs.append(diff_var)
        dict_diff[nmc] = diffs
    return dict_diff


if __name__ == '__main__':
    print("# numpy version:=", np.__version__)
    file_name = './stock.txt'
    str2d = read_file(file_name, 'ASCII')
    portfolio = convert_str_to_float(str2d)
    stock1 = [p[0] for p in portfolio]
    stock2 = [p[1] for p in portfolio]
    port_stocks = {1: stock1, 2: stock2}
    amount1 = 1_000_000.0
    amount2 = 1_500_000.0
    port_amounts = [amount1, amount2]
    stock = stock1.copy()
    amount = amount1
    clevel = 0.95
    days_trading = 251.4
    nmc_set = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    nmc = nmc_set[1]
    is_show = False
    # [Parametric VaR]
    # var_rtn1, var_val1 = VaR_single_asset(stock, amount, clevel, is_show)
    # var_rtnN, var_valN = VaR_multiple_asset(port_stocks, port_amounts, clevel, is_show)

    # plot_gaussian_density(nmc, mu=0.0, sigma=1.0, is_show=True)

    # [Monte Carlo VaR]
    # var_rtn_sim, var_val_sim = MCVaR_single_asset(nmc, stock, amount, clevel, days_trading, is_show)
    # var_rtn_sim, var_val_sim = MCVaR_multiple_asset(nmc, port_stocks, port_amounts, clevel, days_trading, is_show)

    # # [accuracy]
    # is_single = False
    # # nmc_set = nmc_set[:5]
    # nmax = 50
    # if is_single:
    #     dict_diff = check_VaR_accuracy(True, nmc_set, nmax, stock, amount, clevel, days_trading, is_show)
    # else:
    #     dict_diff = check_VaR_accuracy(False, nmc_set, nmax, port_stocks, port_amounts, clevel, days_trading, is_show)
    # for k, v in dict_diff.items():
    #     vr = [round(x,2) for x in v]
    #     print(f'{k}: {vr},')

    dict_diff = {
        # single asset
        # 100: [508.54, 3296.99, 5062.85, 4831.58, 11132.31, 3677.7, 8157.77, 6978.67, 8808.91, 1409.31, 4391.17, 1670.35, 6567.46, 3216.19, 7984.63, 5329.64, 458.57, 1646.6, 8854.36, 9045.01, 5045.51, 5826.49, 2657.33, 1277.48, 768.46, 1520.84, 9087.22, 3672.84, 4388.27, 7255.67, 2180.24, 3624.04, 4550.7, 376.66, 806.07, 4012.89, 8835.1, 6899.26, 96.69, 6773.12, 2206.88, 4422.2, 4678.45, 2964.3, 4342.04, 4003.54, 4730.06, 1800.68, 4608.08, 6879.17],
        # 1000: [771.39, 122.88, 1103.68, 2873.46, 382.71, 2528.06, 287.39, 64.9, 657.4, 151.09, 2070.16, 410.92, 1266.94, 2699.92, 2820.29, 2406.07, 127.63, 857.16, 1158.64, 864.33, 502.94, 2609.64, 56.18, 407.3, 10.29, 2561.83, 330.01, 2226.23, 1672.26, 2115.15, 345.48, 1173.48, 567.4, 1928.12, 1147.53, 406.21, 626.02, 13.75, 163.72, 320.68, 168.12, 2620.52, 587.63, 271.92, 2595.41, 1427.95, 3749.47, 1255.0, 2510.57, 1085.36],
        # 10000: [584.37, 272.24, 500.72, 455.14, 21.94, 158.4, 458.48, 407.77, 937.4, 495.94, 198.07, 379.5, 825.71, 271.38, 273.52, 906.69, 1048.37, 1133.67, 1331.13, 458.71, 862.73, 201.36, 876.38, 138.87, 81.97, 213.68, 212.8, 685.96, 127.22, 1075.06, 15.33, 151.35, 371.4, 382.8, 811.99, 592.8, 1011.5, 691.75, 0.8, 148.85, 892.67, 447.39, 358.92, 7.65, 146.48, 477.21, 872.34, 317.4, 720.06, 634.87],
        # 100000: [427.7, 329.84, 309.5, 115.77, 184.95, 240.59, 6.03, 124.86, 511.4, 383.25, 153.95, 162.24, 171.92, 445.2, 352.18, 36.73, 174.93, 74.3, 388.51, 316.68, 336.44, 522.81, 306.57, 436.79, 303.22, 78.35, 340.96, 211.36, 217.83, 264.41, 205.05, 298.6, 308.01, 398.87, 466.28, 418.12, 238.44, 183.61, 355.15, 543.0, 460.74, 194.05, 455.33, 272.29, 307.61, 390.48, 317.43, 434.12, 424.93, 444.02],
        # 1000000: [219.01, 243.3, 239.17, 247.28, 288.03, 247.72, 255.25, 319.37, 251.63, 309.42, 304.82, 221.78, 250.93, 320.65, 328.46, 315.84, 300.12, 249.39, 253.75, 319.6, 334.54, 359.82, 326.0, 244.93, 329.75, 249.12, 218.48, 320.44, 262.73, 181.45, 211.72, 285.13, 252.02, 230.13, 184.54, 401.07, 228.62, 262.26, 291.15, 270.39, 279.55, 264.84, 263.44, 344.27, 268.76, 201.25, 316.59, 264.2, 210.67, 244.81],
        # 10000000: [280.18, 250.41, 255.08, 297.49, 280.09, 265.94, 271.25, 278.1, 279.35, 259.38, 316.04, 267.21, 265.62, 279.93, 271.43, 270.94, 282.41, 269.7, 273.11, 258.55, 300.22, 283.22, 260.43, 280.38, 274.43, 258.43, 271.86, 298.54, 277.14, 301.47, 273.53, 241.44, 262.11, 291.22, 271.96, 274.08, 256.37, 281.1, 285.22, 258.4, 296.81, 273.18, 272.04, 299.28, 277.05, 272.9, 272.37, 287.91, 291.87, 273.79],
        # multiple asset
        100: [16667.27, 346.0, 619.24, 14922.18, 4607.09, 3942.3, 8054.63, 2275.87, 14551.08, 1227.29, 2080.72, 6150.39, 7602.9, 416.76, 11500.14, 4295.57, 6332.07, 2192.39, 644.81, 15231.45, 696.93, 30340.12, 1956.18, 5022.92, 6909.53, 8654.8, 3916.53, 15979.7, 15408.39, 152.52, 7439.74, 5116.39, 6312.03, 514.47, 6240.38, 13799.69, 9117.69, 14655.65, 7027.21, 119.77, 1329.38, 13221.39, 6620.69, 4218.36, 4175.86, 18643.74, 13076.34, 13458.07, 3804.32, 1378.66],
        1000: [3971.18, 2578.26, 1061.09, 3303.02, 476.09, 4254.7, 3067.9, 1048.65, 4904.65, 3406.12, 934.23, 2488.58, 6247.21, 768.49, 7055.8, 2662.7, 2211.2, 1092.84, 2209.26, 1320.81, 2297.85, 1422.96, 1235.99, 1167.66, 5079.8, 3568.35, 1422.09, 3141.73, 1979.55, 49.14, 175.97, 379.65, 179.93, 467.06, 2580.92, 904.59, 3062.53, 388.97, 3147.98, 680.76, 3797.96, 2635.22, 3642.99, 82.6, 827.98, 125.29, 1088.72, 1148.43, 3166.21, 5107.11],
        10000: [727.51, 308.46, 1274.85, 966.13, 1111.35, 1304.41, 88.25, 160.65, 411.78, 225.27, 618.38, 952.55, 1453.08, 1176.74, 364.11, 600.49, 712.42, 2322.12, 530.57, 1345.35, 328.31, 313.37, 302.55, 1842.7, 1324.92, 2341.62, 1384.86, 332.78, 70.55, 390.8, 118.15, 1487.77, 1082.86, 1359.34, 268.86, 62.79, 318.52, 378.6, 274.73, 192.23, 419.36, 526.72, 1088.52, 388.94, 221.34, 550.66, 316.53, 634.22, 330.57, 124.72],
        100000: [178.56, 381.49, 336.92, 245.51, 364.26, 346.83, 472.09, 325.75, 315.17, 208.21, 338.66, 457.86, 131.17, 152.63, 28.88, 516.14, 344.23, 545.07, 659.59, 61.91, 392.82, 557.32, 224.42, 825.45, 0.11, 596.82, 226.18, 669.0, 183.22, 319.03, 114.26, 563.27, 111.16, 100.44, 174.26, 232.8, 466.04, 758.18, 495.24, 127.58, 406.05, 287.1, 89.55, 144.04, 293.96, 219.35, 156.95, 147.56, 494.58, 22.55],
        1000000: [343.99, 219.32, 264.53, 243.63, 263.36, 358.14, 386.59, 262.12, 168.79, 366.08, 360.01, 289.92, 231.0, 224.11, 299.22, 335.28, 145.18, 355.21, 393.89, 422.91, 316.36, 150.48, 288.97, 308.75, 274.62, 176.24, 207.08, 334.65, 357.0, 220.66, 280.14, 252.91, 358.8, 256.03, 113.29, 264.45, 230.5, 192.41, 235.85, 220.85, 322.12, 338.03, 138.34, 330.85, 207.03, 269.58, 183.65, 301.3, 258.1, 335.82],
        10000000: [321.26, 291.33, 383.71, 289.68, 303.72, 309.95, 290.17, 288.42, 285.76, 322.14, 290.63, 284.26, 298.19, 326.85, 291.64, 303.4, 272.64, 307.45, 302.72, 292.76, 319.3, 306.5, 350.32, 236.15, 276.04, 308.83, 277.28, 258.35, 305.65, 244.75, 296.63, 362.71, 304.63, 285.34, 305.86, 257.11, 250.03, 282.4, 293.62, 299.02, 351.19, 306.32, 297.59, 296.31, 334.32, 255.75, 302.96, 324.98, 290.93, 303.25],
    }
    for k, v in dict_diff.items():
        diff = max(v)-min(v)
        print(k, round(diff,2), round(110000/sqrt(k),2))
    points = [v for k, v in dict_diff.items()]
    fig, ax = plt.subplots()
    bp = ax.boxplot(points)
    ax.set_xticklabels([k for k in dict_diff])
    plt.grid()
    plt.show()

    # ccy = 'USD'
    # dt_fx = read_fx_data('./quote.csv', ccy)
    # rtn_fx = calc_all_return([dt_fx[t] for t in dt_fx])
    # plt.plot(dt_fx.keys(), dt_fx.values(), color='b')
    # xmin = list(dt_fx.keys())[0]
    # xmax = list(dt_fx.keys())[-1]
    # plt.xlim([xmin,xmax])
    # plt.grid(True)
    # plt.show()

