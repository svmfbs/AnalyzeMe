""" Asian option pricing by Vorst92 method
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from math import exp, log, sqrt
import numpy as np
from scipy.stats import norm

# import matplotlib
# matplotlib.rcParams['backend'] = 'QtAgg'
# import matplotlib.pyplot as plt

import warnings
# with warnings.catch_warnings():
warnings.filterwarnings('ignore', category=Warning)

sys.dont_write_bytecode = True # dont make .pyc files


def discount(r: float, tm: float) -> float:
    df = exp(-r*tm)
    return df


class GeometricAsianOption:
    def __init__(self, S0: float, r: float, q: float, vol: float, tm: float, n: int) -> None:
        self.reset_parameters(S0, r, q, vol, tm, n)

    def __set_vol2_hstep(self) -> None:
        self.vol2 = self.vol*self.vol
        self.h = self.tm / self.n

    def reset_parameters(self, S0: float, r: float, q: float, vol: float, tm: float, n: int) -> None:
        assert S0 > 0 and vol > 0 and tm > 0 and n > 0, 'S0 > 0 and vol > 0 and tm > 0 and n > 0 is required'
        self.S0 = S0
        self.r = r
        self.q = q
        self.vol = vol
        self.tm = tm
        self.n = n
        self.__set_vol2_hstep()

    def mean_log_geometric_average(self) -> float:
        """ calculate E[ln(G)]:  G = (\Pi_1^n S_ti)^(1/n)
        """
        rq = self.r - self.q
        mean_lnGeo = log(self.S0) + (rq - 0.5*self.vol2)*(self.tm + self.h) / 2.0
        return mean_lnGeo

    def variance_log_geometric_average(self) -> float:
        """ calculate Var[ln(G)]:  G = (\Pi_1^n S_ti)^(1/n)
        """
        var_lnGeo = self.vol2*(self.h + (self.tm -self.h)*(2*self.n - 1) / (6*self.n))
        return var_lnGeo

    def mean_arithmetic_average(self) -> float:
        """ calculate E[A]:  A = (1/n)\sum_1^n S_ti
        """
        rqh = (self.r - self.q)*self.h
        erqh = exp(rqh)
        erqhn = exp(rqh*self.n)
        mean_arith = (self.S0 / self.n)*erqh*(1.0 - erqhn) / (1.0 - erqh)
        return mean_arith

    def mean_geometric_average(self) -> float:
        """ calculate E[G]:  G = (\Pi_1^n S_ti)^(1/n)
        """
        mean_lnGeo = self.mean_log_geometric_average()
        var_lnGeo = self.variance_log_geometric_average()
        mean_geo = exp(mean_lnGeo + var_lnGeo / 2.0)
        return mean_geo

    def _d1_generic(self, mean_ln: float, K: float, var_ln: float) -> float:
        return (mean_ln - log(K) + var_ln) / sqrt(var_ln)

    def _d2_generic(self, d1: float, var_ln) -> float:
        return d1 - sqrt(var_ln)

    def d1_geometric(self, K: float) -> float:
        """ calculate d1 function of geometric asian option
        """
        mean_lnGeo = self.mean_log_geometric_average()
        var_lnGeo = self.variance_log_geometric_average()
        d1 = self._d1_generic(mean_lnGeo, K, var_lnGeo)
        return d1

    def d2_geometric(self, K: float) -> float:
        """ calculate d2 function of geometric asian option
        """
        var_lnGeo = self.variance_log_geometric_average()
        d1 = self.d1_geometric(K)
        d2 = self._d2_generic(d1, var_lnGeo)
        return d2

    def price_option_call(self, df: float, K: float) -> float:
        """ calculate geometric asian call option: price_call = DF*E[(G-K)^+]
        """
        d1 = self.d1_geometric(K)
        d2 = self.d2_geometric(K)
        mean_geo = self.mean_geometric_average()
        price_geo_call = df*(mean_geo*norm.cdf(d1) - K*norm.cdf(d2))
        return price_geo_call

    def upper_bound_option_call(self, df: float, K: float) -> float:
        """ calculate upper bound of geometric asian call option
        """
        price_geo_call = self.price_option_call(df, K)
        mean_geo = self.mean_geometric_average()
        mean_arith = self.mean_arithmetic_average()
        upbound = price_geo_call + df*(mean_arith - mean_geo)
        return upbound


class ArithmeticAsianOption(GeometricAsianOption):
    def __init__(self, S0: float, r: float, q: float, vol: float, tm: float, n: int) -> None:
        super().__init__(S0, r, q, vol, tm, n)

    def price_option_call_vorst92(self, df: float, K: float) -> float:
        """ calculate arithmetic asian call option approximated by Vorst92 modl: price_call = DF*E[(A-K)^+]
        """
        mean_geo = self.mean_geometric_average()
        mean_arith = self.mean_arithmetic_average()
        Kmod = K - (mean_arith - mean_geo)
        mean_lnGeo = self.mean_log_geometric_average()
        var_lnGeo = self.variance_log_geometric_average()
        d1mod = self._d1_generic(mean_lnGeo, Kmod, var_lnGeo)
        d2mod = self._d2_generic(d1mod, var_lnGeo)
        price_mod_geo_call = df*(mean_geo*norm.cdf(d1mod) - Kmod*norm.cdf(d2mod))
        return price_mod_geo_call


if __name__ == '__main__':
    msg = '# Asian option pricing by Vorst92 method'
    print(msg)
    S0 = 100.0
    r = 0.05
    q = 0.0
    #
    Kc = 100.0
    vc = 0.20
    tc = 1.0
    nc = 12
    Ks = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0]
    vols = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    ts = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    ns = [12, 24, 36, 48, 60]
    ndp = 2
    results_v = [2.93, 3.90, 5.00, 6.13, 7.27, 8.40, 9.53, 10.66, 11.77, 12.86]
    results_t = [4.10, 6.13, 7.78, 9.24, 10.55, 11.76, 12.88, 13.93, 14.91, 15.85]
    results_n = [6.13, 5.93, 5.86, 5.83, 5.81]
    results_s = [31.16, 26.42, 21.72, 17.17, 12.92, 9.18, 6.13, 3.82, 2.23, 1.22, 0.63, 0.30, 0.14]
    asian_arith = ArithmeticAsianOption(S0, r, q, vc, tc, nc)
    print('# ------------------------------------------')
    for i, v in enumerate(vols):
        df = discount(r, tc)
        asian_arith.reset_parameters(S0, r, q, v, tc, nc)
        price_vorst = asian_arith.price_option_call_vorst92(df, Kc)
        price_ubound = asian_arith.upper_bound_option_call(df, Kc)
        price_geo = asian_arith.price_option_call(df, Kc)
        price_dp = round(price_vorst, ndp)
        res = results_v[i]
        diff = price_dp - res
        print(f'{i}, {v}, {Kc}, {nc}, {tc}, {price_dp}, {diff}, {round(price_ubound, ndp)}, {round(price_geo, ndp)}')
    print('# ------------------------------------------')
    for i, t in enumerate(ts):
        df = discount(r, t)
        asian_arith.reset_parameters(S0, r, q, vc, t, nc)
        price_vorst = asian_arith.price_option_call_vorst92(df, Kc)
        price_ubound = asian_arith.upper_bound_option_call(df, Kc)
        price_geo = asian_arith.price_option_call(df, Kc)
        price_dp = round(price_vorst, ndp)
        res = results_t[i]
        diff = price_dp - res
        print(f'{i}, {vc}, {Kc}, {nc}, {t}, {price_dp}, {diff}, {round(price_ubound, ndp)}, {round(price_geo, ndp)}')
    print('# ------------------------------------------')
    for i, n in enumerate(ns):
        df = discount(r, tc)
        asian_arith.reset_parameters(S0, r, q, vc, tc, n)
        price_vorst = asian_arith.price_option_call_vorst92(df, Kc)
        price_ubound = asian_arith.upper_bound_option_call(df, Kc)
        price_geo = asian_arith.price_option_call(df, Kc)
        price_dp = round(price_vorst, ndp)
        res = results_n[i]
        diff = price_dp - res
        print(f'{i}, {vc}, {Kc}, {n}, {tc}, {price_dp}, {diff}, {round(price_ubound, ndp)}, {round(price_geo, ndp)}')
    print('# ------------------------------------------')
    asian_arith.reset_parameters(S0, r, q, vc, tc, nc)
    for i, K in enumerate(Ks):
        df = discount(r, tc)
        price_vorst = asian_arith.price_option_call_vorst92(df, K)
        price_ubound = asian_arith.upper_bound_option_call(df, K)
        price_geo = asian_arith.price_option_call(df, K)
        price_dp = round(price_vorst, ndp)
        res = results_s[i]
        diff = price_dp - res
        print(f'{i}, {vc}, {K}, {nc}, {tc}, {price_dp}, {diff}, {round(price_ubound, ndp)}, {round(price_geo, ndp)}')
    print('# ------------------------------------------')
