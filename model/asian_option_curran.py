""" Asian option pricing by Curran94 method
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from math import exp, log, sqrt
import numpy as np
from scipy.stats import norm
from hep.model.asian_option_vorst import discount, GeometricAsianOption

# import matplotlib
# matplotlib.rcParams['backend'] = 'QtAgg'
# import matplotlib.pyplot as plt

import warnings
# with warnings.catch_warnings():
warnings.filterwarnings('ignore', category=Warning)

sys.dont_write_bytecode = True # dont make .pyc files


class ArithmeticAsianOptionGeometricCondition(GeometricAsianOption):
    _mean_lnSi = []
    _var_lnSi = []
    _covar_lnSi_lnG = []
    _stepsize = 1e-6
    _lower_limit = 0.0

    def __init__(self, S0: float, r: float, q: float, vol: float, tm: float, n: int) -> None:
        super().__init__(S0, r, q, vol, tm, n)
        self.mean_lnGeo = self.mean_log_geometric_average()
        self.var_lnGeo = self.variance_log_geometric_average()
        self.set_mean_var_covar_lnSi_lnG()

    @property
    def lower_limit_C1(self) -> float:
        if self._lower_limit > 0.0:
            return self._lower_limit
        else:
            raise ValueError('# error: not yet calculated')

    def reset_parameters(self, S0: float, r: float, q: float, vol: float, tm: float, n: int) -> None:
        super().reset_parameters(S0, r, q, vol, tm, n)
        self.mean_lnGeo = self.mean_log_geometric_average()
        self.var_lnGeo = self.variance_log_geometric_average()
        self.set_mean_var_covar_lnSi_lnG()

    def __get_mean_var_lnGeo(self) -> tuple[float, float]:
        return self.mean_lnGeo, self.var_lnGeo

    def variance_log_geometric_average(self) -> float:
        """ calculate Var[ln(G)]:  G = (\Pi_1^n S_ti)^(1/n)
        NOTE: same as vorst92, but expression is different
        """
        var_lnGeo = self.vol2*self.h*(2.0*self.n + 1.0)*(self.n + 1.0) / (6.0*self.n)
        return var_lnGeo

    def set_mean_var_covar_lnSi_lnG(self) -> None:
        """ set mean, variance of ln(Si) and covariance between ln(Si) and ln(G)
        \mu_i    := E[ln(S_i)]
        \sig^2_i := Var[ln(S_i)]
        \gamma_i := Covar[ln(S_i), ln(G)]
        """
        self._mean_lnSi.clear()
        self._var_lnSi.clear()
        self._covar_lnSi_lnG.clear()
        nend = self.n + 1
        for i1 in range(1, nend):
            i2 = i1 * i1
            ih = i1 * self.h
            rq = self.r - self.q
            self._mean_lnSi.append(log(self.S0) + (rq - 0.5*self.vol2)*ih)
            self._var_lnSi.append(self.vol2*ih)
            self._covar_lnSi_lnG.append(self.vol2*self.h*((2.0*self.n + 1.0)*i1 - i2) / (2.0*self.n))
        return

    def mean_arithmetic_average_conditioning_geometric(self, ln_x: float) -> float:
        """ calculate expected value of arithmetic price conditioning geometric price
        E[A|G=x] = E[A|ln(G)=ln(x)]
        """
        # muG = self.mean_lnGeo
        # varG = self.var_lnGeo
        muG, varG = self.__get_mean_var_lnGeo()
        y_muG = ln_x - muG
        result = 0.0
        for i0 in range(self.n):
            mui = self._mean_lnSi[i0]
            vari = self._var_lnSi[i0]
            gammai = self._covar_lnSi_lnG[i0]
            gammai2 = gammai*gammai
            term1 = mui + y_muG*gammai / varG
            term2 = 0.5*(vari - gammai2 / varG)
            result += exp(term1 + term2)
        cond_value = result / self.n
        return cond_value

    def __calc_integralC1_lower_limit(self, K: float) -> float:
        """ calculate lower limit of C1 integral
        L := {x | E[A|G=x] = K} = {x | E[A|ln(G)=ln(x)] = K}
        """
        hs = self._stepsize
        y = log(K)
        eAG = self.mean_arithmetic_average_conditioning_geometric(y)
        while eAG > K:
            y = y - hs
            eAG = self.mean_arithmetic_average_conditioning_geometric(y)
        # at this point, eAG < K, so we use the last but one y
        y = y + hs
        self._lower_limit = exp(y)
        return self._lower_limit

    def price_option_call_curran94(self, df: float, K: float) -> float:
        """ calculate arithmetic asian call option approximated by Curran94 modl: price_call = DF*E[(A-K)^+]
        """
        muG, varG = self.__get_mean_var_lnGeo()
        L = self.__calc_integralC1_lower_limit(K)
        c1 = 0.0
        for i0 in range(self.n):
            mui = self._mean_lnSi[i0]
            vari = self._var_lnSi[i0]
            gammai = self._covar_lnSi_lnG[i0]
            d1 = self._d1_generic(muG, L, gammai, varG)
            c1 += exp(mui + 0.5*vari)*norm.cdf(d1)
        c1 = c1 / self.n
        d2 = self._d1_generic(muG, L, 0.0, varG)
        c2 = K * norm.cdf(d2)
        price = df*(c1 - c2)
        return price


if __name__ == '__main__':
    msg = '# Asian option pricing by Curran94 method'
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
    results_v = [2.93, 3.90, 5.01, 6.16, 7.31, 8.47, 9.64, 10.80, 11.96, 13.12]
    results_t = [4.11, 6.16, 7.84, 9.32, 10.66, 11.90, 13.05, 14.14, 15.16, 16.13]
    results_n = [6.16, 5.96, 5.89, 5.86, 5.84]
    results_s = [31.16, 26.42, 21.72, 17.16, 12.92, 9.19, 6.16, 3.87, 2.29, 1.28, 0.67, 0.34, 0.16]
    curran = ArithmeticAsianOptionGeometricCondition(S0, r, q, vols[0], tc, nc)
    print('# ------------------------------------------')
    for i, v in enumerate(vols):
        df = discount(r, tc)
        curran.reset_parameters(S0, r, q, v, tc, nc)
        price = curran.price_option_call_curran94(df, Kc)
        price_dp = round(price, ndp)
        res = results_v[i]
        diff = price_dp - res
        limit = round(curran.lower_limit_C1, 4)
        print(f'{i}, {v}, {Kc}, {nc}, {tc}, {price_dp}, {diff}, {limit}')
    print('# ------------------------------------------')
    for i, t in enumerate(ts):
        df = discount(r, t)
        curran.reset_parameters(S0, r, q, vc, t, nc)
        price = curran.price_option_call_curran94(df, Kc)
        price_dp = round(price, ndp)
        res = results_t[i]
        diff = price_dp - res
        limit = round(curran.lower_limit_C1, 4)
        print(f'{i}, {vc}, {Kc}, {nc}, {t}, {price_dp}, {diff}, {limit}')
    print('# ------------------------------------------')
    for i, n in enumerate(ns):
        df = discount(r, tc)
        curran.reset_parameters(S0, r, q, vc, tc, n)
        price = curran.price_option_call_curran94(df, Kc)
        price_dp = round(price, ndp)
        res = results_n[i]
        diff = price_dp - res
        limit = round(curran.lower_limit_C1, 4)
        print(f'{i}, {vc}, {Kc}, {n}, {tc}, {price_dp}, {diff}, {limit}')
    print('# ------------------------------------------')
    curran.reset_parameters(S0, r, q, vc, tc, nc)
    for i, K in enumerate(Ks):
        df = discount(r, tc)
        price = curran.price_option_call_curran94(df, K)
        price_dp = round(price, ndp)
        res = results_s[i]
        diff = price_dp - res
        limit = round(curran.lower_limit_C1, 4)
        print(f'{i}, {vc}, {K}, {nc}, {tc}, {price_dp}, {diff}, {limit}')
    print('# ------------------------------------------')
