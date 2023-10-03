""" American Option Pricing
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from math import sqrt
from scipy.stats import norm
import numpy as np

sys.dont_write_bytecode = True # dont make .pyc files


def N(x: float):
    """ cumulative normal distribution function
    """
    return norm.cdf(x)


def N_prime(x: float):
    """ standard normal probability density function
    """
    return norm.pdf(x)


def d1BS(spot: float, strike: float, rate: float, qrate: float, vol: float, expiry: float) -> float:
    """ d1 in Black-Scholes
    """
    brq = rate - qrate
    rqT = brq*expiry
    vsT = vol*np.sqrt(expiry)
    tup = np.log(spot / strike) + (rqT + 0.5*vsT*vsT)
    d1_func = tup / vsT
    return d1_func


def d2BS(spot: float, strike: float, rate: float, qrate: float, vol: float, expiry: float) -> float:
    """ d2 in Black-Scholes
    """
    vsT = vol*np.sqrt(expiry)
    d1_func = d1BS(spot, strike, rate, qrate, vol, expiry)
    d2_func = d1_func - vsT
    return d2_func


def price_european(is_call: bool, DF: float, spot: float, strike: float, rate: float, qrate: float, vol: float, expiry: float) -> float:
    """ discounted call/put option price in Black-Scholes formula
    """
    flag = 1.0 if is_call else -1.0
    d1 = d1BS(spot, strike, rate, qrate, vol, expiry)
    d2 = d2BS(spot, strike, rate, qrate, vol, expiry)
    brq = rate - qrate
    DFb = np.exp(brq * expiry)
    term1 = spot * DFb * N(flag*d1)
    term2 = strike * N(flag*d2)
    return flag*DF*(term1 - term2)

# #################################################
# ##### [American option pricing] #################
# #################################################

def Mfunc(rate: float, vol: float) -> float:
    vol2 = vol*vol
    return 2.0*rate/vol2


def Nfunc(brate: float, vol: float) -> float:
    vol2 = vol*vol
    return 2.0*brate/vol2


def Kfunc(rate: float, tau: float) -> float:
    return 1.0 - np.exp(-rate*tau)


def func_N_M_K(rate: float, brate: float, vol: float, tau: float) -> tuple[float, float, float]:
    N = Nfunc(brate, vol)
    M = Mfunc(rate, vol)
    K = Kfunc(rate, tau)
    return (N, M, K)


def q1func(rate: float, brate: float, vol: float, tau: float) -> float:
    N, M, K = func_N_M_K(rate, brate, vol, tau)
    b = N-1
    b2 = b*b
    d = 4.0*M/K
    return 0.5*(-b - sqrt(b2 + d))


def q2func(rate: float, brate: float, vol: float, tau: float) -> float:
    N, M, K = func_N_M_K(rate, brate, vol, tau)
    b = N-1
    b2 = b*b
    d = 4.0*M/K
    return 0.5*(-b + sqrt(b2 + d))


def calc_DF_b_r(rate: float, brate: float, tau: float) -> float:
    return np.exp((brate-rate)*tau)


def A1(spot_critical: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ A1 must be greater than zero since q1 < 0 and critical spot price > 0.
    """
    brDF = calc_DF_b_r(rate, brate, tau)
    qrate = rate - brate
    d1f = d1BS(spot_critical, strike, rate, qrate, vol, tau)
    N1m = N(-d1f)
    q1f = q1func(rate, brate, vol, tau)
    return -(spot_critical/q1f)*(1.0 - brDF*N1m)


def A2(spot_critical: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ A2 must be greater than zero since q2 > 0 and critical spot price > 0 (b < r).
    """
    brDF = calc_DF_b_r(rate, brate, tau)
    qrate = rate - brate
    d1f = d1BS(spot_critical, strike, rate, qrate, vol, tau)
    N1p = N(d1f)
    q2f = q2func(rate, brate, vol, tau)
    return (spot_critical/q2f)*(1.0 - brDF*N1p)


def call_bslope(spot: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ calculate b-slope of call option for the iterative solver
    """
    brDF = calc_DF_b_r(rate, brate, tau)
    qrate = rate - brate
    d1f = d1BS(spot, strike, rate, qrate, vol, tau)
    N1p = N(d1f)
    phi = N_prime(d1f)
    vsT = vol*np.sqrt(tau)
    q2f = q2func(rate, brate, vol, tau)
    term1 = brDF*N1p*(1.0 - 1.0/q2f)
    term2 = (1.0 - brDF*phi/vsT)/q2f
    return term1 + term2


def call_LHS(spot: float, strike: float) -> float:
    """ Left hand side of the boundary touch equation for call
    """
    return spot - strike


def call_RHS(spot: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ Right hand side of the boundary touch equation for call
    """
    DF = np.exp(-rate*tau)
    qrate = rate - brate
    prem_call = price_european(True, DF, spot, strike, rate, qrate, vol, tau)
    brDF = calc_DF_b_r(rate, brate, tau)
    d1f = d1BS(spot, strike, rate, qrate, vol, tau)
    N1p = N(d1f)
    q2f = q2func(rate, brate, vol, tau)
    term = (1.0 - brDF*N1p)*spot/q2f
    return prem_call + term


def call_spot_next(spot: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ next candidate spot price for call
    """
    rhs = call_RHS(spot, strike, rate, brate, vol, tau)
    bsi = call_bslope(spot, strike, rate, brate, vol, tau)
    term_up = strike + rhs - bsi*spot
    term_dn = 1.0 - bsi
    return term_up / term_dn


def put_bslope(spot: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ calculate b-slope of put option for the iterative solver
    """
    brDF = calc_DF_b_r(rate, brate, tau)
    qrate = rate - brate
    d1f = d1BS(spot, strike, rate, qrate, vol, tau)
    N1m = N(-d1f)
    phi = N_prime(d1f)
    vsT = vol*np.sqrt(tau)
    q1f = q1func(rate, brate, vol, tau)
    term1 = brDF*N1m*(1.0 - 1.0/q1f)
    term2 = (1.0 + brDF*phi/vsT)/q1f
    return -term1 - term2


def put_LHS(spot: float, strike: float) -> float:
    """ Left hand side of the boundary touch equation for put
    """
    return strike - spot


def put_RHS(spot: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ Right hand side of the boundary touch equation for call
    """
    DF = np.exp(-rate*tau)
    qrate = rate - brate
    prem_put = price_european(False, DF, spot, strike, rate, qrate, vol, tau)
    brDF = calc_DF_b_r(rate, brate, tau)
    d1f = d1BS(spot, strike, rate, qrate, vol, tau)
    N1m = N(-d1f)
    q1f = q1func(rate, brate, vol, tau)
    term = (1.0 - brDF*N1m)*spot/q1f
    return prem_put - term


def put_spot_next(spot: float, strike: float, rate: float, brate: float, vol: float, tau: float) -> float:
    """ next candidate spot price for put
    """
    rhs = put_RHS(spot, strike, rate, brate, vol, tau)
    bsi = put_bslope(spot, strike, rate, brate, vol, tau)
    term_up = strike - rhs + bsi*spot
    term_dn = 1.0 + bsi
    return term_up / term_dn


def calc_critical_spot(is_call: bool, strike: float, rate: float, brate: float, vol: float, tau: float, niter: int) -> tuple[float, int]:
    """ calculate critical spot prices, S^* for call, S^** for put
    """
    eps = 0.00001
    spot = strike
    if is_call:
        for nc in range(niter):
            lhs_call = call_LHS(spot, strike)
            rhs_call = call_RHS(spot, strike, rate, brate, vol, tau)
            diff_call = abs(lhs_call - rhs_call)/strike
            if diff_call < eps:
                return (spot, nc)
            else:
                spot = call_spot_next(spot, strike, rate, brate, vol, tau)
    else:
        for np in range(niter):
            lhs_put = put_LHS(spot, strike)
            rhs_put = put_RHS(spot, strike, rate, brate, vol, tau)
            diff_put = abs(lhs_put - rhs_put)/strike
            if diff_put < eps:
                return (spot, np)
            else:
                spot = put_spot_next(spot, strike, rate, brate, vol, tau)
    raise ValueError(f'# iteration failure: {niter}')


def calc_early_exercise_prem(is_call: bool, spot: float, strike: float, rate: float, brate: float, vol: float, tau: float, spot_critical: float) -> float:
    """ early exercise premium which is defined as american option premium - european option premium
    """
    assert spot_critical > 0, 'spot_critical > 0 is required'
    if is_call:
        q2f = q2func(rate, brate, vol, tau)
        A2f = A2(spot_critical, strike, rate, brate, vol, tau)
        return A2f*pow(spot/spot_critical, q2f)
    else:
        q1f = q1func(rate, brate, vol, tau)
        A1f = A1(spot_critical, strike, rate, brate, vol, tau)
        return A1f*pow(spot/spot_critical, q1f)


def price_american_call(DF: float, spot: float, strike: float, rate: float, brate: float, vol: float, tau: float, spot_critical: float) -> float:
    """ american call option with quadaratic approximation
    """
    if spot >= spot_critical:
        return spot - strike
    qrate = rate - brate
    prem_euro = price_european(True, DF, spot, strike, rate, qrate, vol, tau)
    if brate >= rate:
        return prem_euro
    else:
        prem_early_exercise = calc_early_exercise_prem(True, spot, strike, rate, brate, vol, tau, spot_critical)
        return prem_euro + prem_early_exercise


def price_american_put(DF: float, spot: float, strike: float, rate: float, brate: float, vol: float, tau: float, spot_critical: float) -> float:
    """ american put option with quadaratic approximation
    """
    if spot <= spot_critical:
        return strike - spot
    qrate = rate - brate
    prem_euro = price_european(False, DF, spot, strike, rate, qrate, vol, tau)
    prem_early_exercise = calc_early_exercise_prem(False, spot, strike, rate, brate, vol, tau, spot_critical)
    return prem_euro + prem_early_exercise


def price_american(is_call: bool, DF: float, spot: float, strike: float, rate: float, brate: float, vol: float, tau: float, spot_critical: float) -> float:
    """ american call/put option with quadaratic approximation (r > 0)
    """
    if is_call:
        return price_american_call(DF, spot, strike, rate, brate, vol, tau, spot_critical)
    else:
        return price_american_put(DF, spot, strike, rate, brate, vol, tau, spot_critical)


def results_american(is_call: bool, itable: int, iparam: int) -> list[float]:
    """ results for table I,II,III and V
    """
    if is_call:
        prem_call = {
            # table.1
            (1, 1): [0.03, 0.59, 3.52, 10.31, 20.0],
            (1, 2): [0.03, 0.59, 3.51, 10.29, 20.0],
            (1, 3): [1.07, 3.28, 7.41, 13.50, 21.23],
            (1, 4): [0.23, 1.39, 4.72, 10.96, 20.00],
            # table.2
            (2, 1): [0.05, 0.85, 4.44, 11.66, 20.90],
            (2, 2): [0.05, 0.84, 4.40, 11.55, 20.69],
            (2, 3): [1.29, 3.82, 8.35, 14.80, 22.72],
            (2, 4): [0.41, 2.18, 6.50, 13.42, 22.06],
            # table.3
            (3, 1): [0.04, 0.70, 3.93, 10.81, 20.02],
            (3, 2): [0.04, 0.70, 3.90, 10.75, 20.00],
            (3, 3): [1.17, 3.53, 7.84, 14.08, 21.86],
            (3, 4): [0.30, 1.72, 5.48, 11.90, 20.34],
            # table.5
            (5, 1): [2.52,  4.97,  8.67, 13.88, 20.88],
            (5, 2): [4.20,  7.54, 12.03, 17.64, 24.30],
            (5, 3): [6.97, 11.62, 17.40, 24.09, 31.49],
        }
        return prem_call[(itable, iparam)]
    else:
        prem_put = {
            # table.1
            (1, 1): [20.42, 11.25, 4.40, 1.12, 0.18],
            (1, 2): [20.25, 11.15, 4.35, 1.11, 0.18],
            (1, 3): [21.46, 13.93, 8.27, 4.52, 2.30],
            (1, 4): [20.98, 12.64, 6.37, 2.65, 0.92],
            # table.2
            (2, 1): [20.00, 10.18, 3.54, 0.80, 0.12],
            (2, 2): [20.00, 10.16, 3.53, 0.79, 0.12],
            (2, 3): [20.53, 12.93, 7.46, 3.96, 1.95],
            (2, 4): [20.00, 10.71, 4.77, 1.76, 0.55],
            # table.3
            (3, 1): [20.00, 10.58, 3.93, 0.94, 0.15],
            (3, 2): [20.00, 10.53, 3.90, 0.93, 0.15],
            (3, 3): [20.93, 13.39, 7.84, 4.23, 2.12],
            (3, 4): [20.04, 11.48, 5.48, 2.15, 0.70],
            # table.5
            (5, 1): [26.25, 20.64, 15.99, 12.22,  9.23],
            (5, 2): [22.40, 16.50, 12.03,  8.69,  6.22],
            (5, 3): [20.33, 13.56,  9.11,  6.12,  4.12],
            (5, 4): [20.00, 11.63,  6.96,  4.26,  2.64],
        }
        return prem_put[(itable, iparam)]


class DEMOAmericanOptionQuadratic:
    def __init__(self) -> None:
        pass

    @classmethod
    def __execute(cls, itable: int, iparam: int, is_call: bool, DF: float, Spots: list[float], X: float, r: float, b: float, vol: float, T: float, niter):
        q = r - b
        S = Spots[2]
        print('# European/American option pricing')
        print(f'# b={b}, X={X}, r={r}, q={q}, vol={vol}, T={T}, S={S}')
        spot_c, niter = calc_critical_spot(is_call, X, r, b, vol, T, niter)
        print(f'is call:= {is_call}')
        print(f'critical spot:= {spot_c:.3f}')
        print(f'iteration num:= {niter}')
        answ_amer = results_american(is_call, itable, iparam)
        for i, s in enumerate(Spots):
            prem_euro = price_european(is_call, DF, s, X, r, q, vol, T)
            prem_amer = price_american(is_call, DF, s, X, r, b, vol, T, spot_c)
            diff_b0 = round(prem_amer, 2) - answ_amer[i]
            print(f'Price S:={s}, European call:= {round(prem_euro, 2)}, American call:= {round(prem_amer, 2)}, diff:= {diff_b0:.3f}')

    @classmethod
    def main_table123(cls, msg: str, is_call: bool, itable: int, iparam: int):
        print(msg)
        brates = {
            1: -0.04,
            2: 0.04,
            3: 0.00,
        }
        b = brates[itable]
        option_params_r_vol_T = {
            1: (0.08, 0.20, 0.25),
            2: (0.12, 0.20, 0.25),
            3: (0.08, 0.40, 0.25),
            4: (0.08, 0.20, 0.50),
        }
        r, vol, T = option_params_r_vol_T[iparam]
        Spots = [80.0, 90.0, 100.0, 110.0, 120.0]
        X = 100.0
        DF = np.exp(-r*T)
        niter = 50
        cls.__execute(itable, iparam, is_call, DF, Spots, X, r, b, vol, T, niter)

    @classmethod
    def main_table5(cls, msg: str, is_call: bool, iparam: int):
        print(msg)
        option_params_b_r_vol_T = {
            1: (-0.04, 0.08, 0.20, 3.00),
            2: ( 0.00, 0.08, 0.20, 3.00),
            3: ( 0.04, 0.08, 0.20, 3.00),
            4: ( 0.08, 0.08, 0.20, 3.00),
        }
        b, r, vol, T = option_params_b_r_vol_T[iparam]
        Spots = [80.0, 90.0, 100.0, 110.0, 120.0]
        X = 100.0
        DF = np.exp(-r*T)
        niter = 50
        itable = 5
        cls.__execute(itable, iparam, is_call, DF, Spots, X, r, b, vol, T, niter)


if __name__ == '__main__':
    msg = '# American option pricing with quadratic approximation'
    is_call = False
    is_table123 = False
    if is_table123:
        for itable in [1, 2, 3]:
            for iparam in [1, 2, 3, 4]:
                DEMOAmericanOptionQuadratic.main_table123(msg, is_call, itable, iparam)
            print()
    else:
        for iparam in [1, 2, 3, 4]:
            if is_call:
                if iparam == 4: break
            DEMOAmericanOptionQuadratic.main_table5(msg, is_call, iparam)
