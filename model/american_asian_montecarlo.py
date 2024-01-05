""" American Asian Option Pricing
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from math import exp
import numpy as np
# import matplotlib.pyplot as plt
import warnings
# with warnings.catch_warnings():
warnings.filterwarnings('ignore', category=Warning)
from hep.model.least_square_method import BasisType, NPLeastSquareSimple
from hep.model.american_option_montecarlo import GeneratePriceProcessSimple, CashflowPayoff

sys.dont_write_bytecode = True # dont make .pyc files


class OptionAmericanAsianMonteCarlo(CashflowPayoff):
    _epsilon_lowest = 1.0e-128
    _epsilon = 0.0000000001
    _last_cashflows: list[float]
    _all_cashflows: dict[int, list[float]]
    _all_regcoeffs: dict[int, np.ndarray]
    _min_time_index: int

    def __init__(self, avg_spot: float, spot_price: float, strike: float, t_maturity: float, rate: float, qrate: float, spot_paths: np.ndarray) -> None:
        super().__init__(strike)
        self.avg_spot = avg_spot
        self.spot = spot_price
        self.t_maturity = t_maturity
        self.rate = rate
        self.qate = qrate
        self.brate = rate - qrate
        self.reset_spot_paths(spot_paths)
        itend = self.ntimesM1
        self._all_regcoeffs = {itend: np.zeros(0)}

    def reset_spot_paths(self, spot_paths: np.ndarray) -> None:
        for spot_init in spot_paths[0]:
            diff_spot_init = self.spot - spot_init
            assert abs(diff_spot_init) < self._epsilon, f'{abs(diff_spot_init) < self._epsilon} is required'
        self.spot_paths = spot_paths
        self.ntimes, self.mpaths = spot_paths.shape
        assert self.ntimes > 1, f'{self.ntimes} > 1 is required'
        assert self.mpaths > 1, f'{self.mpaths} > 1 is required'
        itend = self.ntimes - 1
        self.ntimesM1 = itend
        self.deltaT = self.t_maturity / self.ntimesM1
        print(f'# ntimes  := {self.ntimes}, mpaths:= {self.mpaths}')
        print(f'# ntimesM1:= {self.ntimesM1}, deltaT:= {self.deltaT}, expiry:= {self.deltaT*self.ntimesM1}')
        self.zeros = [0.0]*self.mpaths

    def show_spot_paths(self, ndp: int=4) -> None:
        print('# original spot price paths ----------------- ')
        for itime in reversed(range(self.ntimes)):
            print(f'{itime}: ', end='')
            for jmc in range(self.mpaths):
                print(f'{round(self.spot_paths[itime, jmc], ndp)}, ', end='')
            print()
        print()

    def show_average_spot_paths(self, ndp: int=4) -> None:
        print('# original average spot price paths --------- ')
        for itime in reversed(range(self.ntimesM1)):
            print(f'{itime}: ', end='')
            for jmc in range(self.mpaths):
                print(f'{round(self.avgs_paths[itime, jmc], ndp)}, ', end='')
            print()
        print()

    def show_cashflows(self, ndp: int=4) -> None:
        print('# result all discounted cashflows ----------- ')
        if len(self._all_cashflows) > 0:
            for k, row in self._all_cashflows.items():
                print(f'{k:3d}, {[round(r, ndp) for r in row]}')
        else:
            print('# error: no cashflows')
        print()

    def show_regcoeffs(self, ndp: int=4) -> None:
        print('# result all regression coefficients ----------- ')
        if len(self._all_regcoeffs) > 0:
            for k, row in self._all_regcoeffs.items():
                print(f'{k:3d}, {[round(a, ndp) for a in np.ravel(row)]}')
        else:
            print('# error: no regression coefficients')
        print()

    def discount(self, rate: float, deltaT: float) -> float:
        return exp(-rate*deltaT)

    def __set_last_payoffs(self, is_call: bool, avg_delta: float) -> None:
        """ set payoffs at last expiry, put them into all cashflows
        """
        itend = self.ntimesM1
        mpaths = self.mpaths
        avgs0 = self.avg_spot
        deltaT = self.deltaT
        init_sum_spot = avg_delta*avgs0
        sum_spots = np.empty(mpaths)
        self.avgs_paths = np.zeros(self.spot_paths.shape)
        self._all_cashflows = {0: self.make_cashflows_at(is_call, [avgs0 for _ in range(mpaths)])}
        for it_curr in range(self.ntimesM1):
            it_next = it_curr + 1
            time = it_curr * deltaT
            total_time = time + avg_delta
            for jm in range(mpaths):
                spot_curr = self.spot_paths[it_curr, jm]
                spot_next = self.spot_paths[it_next, jm]
                sbar_next = 0.5*(spot_curr + spot_next)
                sum_spots[jm] += sbar_next * deltaT
            avg_prices = np.array([(su + init_sum_spot) / total_time for su in sum_spots])
            self.avgs_paths[it_next] = avg_prices
            self._all_cashflows[it_next] = self.make_cashflows_at(is_call, avg_prices)
        self._last_cashflows = self._all_cashflows[itend][:]

    def __max_cashflows_discounted_backward(self, it_prev: int, dfdt: float, exercise_paths: list[int], curr_cashflows: list[float]) -> list[float]:
        cmax_cashflows = [dfdt*cfprev for cfprev in self._all_cashflows[it_prev]]
        for ie in exercise_paths:
            cmax_cashflows[ie] = curr_cashflows[ie]
        return cmax_cashflows

    def __max_cashflows_nodiscount_foreward(self, exercise_paths: list[int], curr_cashflows: list[float]) -> list[float]:
        cmax_cashflows = self.zeros[:]
        for ie in exercise_paths:
            cmax_cashflows[ie] = curr_cashflows[ie]
        return cmax_cashflows

    def store_all_cashflows_backward(self, is_call: bool, basis: BasisType, order_basis: int, t_lockout: float, avg_delta: float) -> None:
        """ calculate all cashflows with optimal early exercise from backward
        """
        print('# AMC backward -----')
        self.is_backward = True
        self.t_lockout = t_lockout
        self.avg_delta = avg_delta
        ntimesM1 = self.ntimesM1
        # mpaths = self.mpaths
        rate_c = self.rate
        deltaT = self.deltaT
        print(f'deltaT:= {deltaT}, t_lockout:= {t_lockout}')
        self.__set_last_payoffs(is_call, avg_delta)
        self._min_time_index = 0
        for it_curr in reversed(range(1, ntimesM1)):
            it_prev = it_curr + 1 # do not move into if-statement
            time = it_curr * deltaT
            if time >= t_lockout:
                dfdt = self.discount(rate_c, deltaT)
                spot_paths_curr = self.spot_paths[it_curr, :]
                avgs_paths_curr = self.avgs_paths[it_curr, :]
                curr_cashflows = self._all_cashflows[it_curr]
                itm_cashflows = self.itm_path_cashflows(curr_cashflows)
                x1s = [spot_paths_curr[itm] for itm in itm_cashflows]
                x2s = [avgs_paths_curr[itm] for itm in itm_cashflows]
                ys = [dfdt * self._all_cashflows[it_prev][itm] for itm in itm_cashflows]
                # [regression]
                # design_matrix = NPLeastSquareSimple.design_matrix_cubic_model_2vars(x1s, x2s)
                design_matrix = NPLeastSquareSimple.design_matrix_cubic_general_model_2vars(basis, x1s, x2s)
                if len(x1s) + len(x2s) < order_basis:
                    print(f'# [warn]: {it_curr}, len(x1s){len(x1s)} + len(x2s){len(x2s)} < order_basis{order_basis}')
                    mcols = design_matrix.shape[1]
                    avec0 = np.array([[0.0] for _ in range(mcols)])
                else:
                    avec0 = NPLeastSquareSimple.beta_coefficients(design_matrix, np.array(ys))
                self._all_regcoeffs[it_curr] = avec0
                ycnd_expected = [np.dot(avec0, xr) for xr in design_matrix]
                assert len(itm_cashflows) == len(ycnd_expected), f'{len(itm_cashflows)} == {len(ycnd_expected)} is required'
                exercise_judge = {itm: itm_cashflows[itm] > ycnd_expected[k] for k, itm in enumerate(itm_cashflows)}
                exercise_paths = [itm for itm in exercise_judge if exercise_judge[itm]]
                cmax_cashflows = self.__max_cashflows_discounted_backward(it_prev, dfdt, exercise_paths, curr_cashflows)
                self._all_cashflows[it_curr] = cmax_cashflows[:]
                self._all_cashflows[it_prev] = self.zeros[:]
            else:
                self._min_time_index = it_prev
                for i in range(it_prev):
                    self._all_cashflows[i] = self.zeros[:]
                break

    def store_all_cashflows_foreward(self, is_call: bool, basis: BasisType, order_basis: int) -> None:
        """ calculate all cashflows from foreward with regressions
        """
        print('# AMC foreward -----')
        self.is_backward = False
        t_lockout = self.t_lockout
        avg_delta = self.avg_delta
        ntimesM1 = self.ntimesM1
        # mpaths = self.mpaths
        # rate_c = self.rate
        deltaT = self.deltaT
        all_exercise_paths: set[int] = set()
        self.__set_last_payoffs(is_call, avg_delta)
        for it_curr in range(1, ntimesM1):
            time = it_curr * deltaT
            if time >= t_lockout:
                spot_paths_curr = self.spot_paths[it_curr, :]
                avgs_paths_curr = self.avgs_paths[it_curr, :]
                curr_cashflows = self._all_cashflows[it_curr]
                itm_cashflows = self.itm_path_cashflows(curr_cashflows)
                x1s = [spot_paths_curr[itm] for itm in itm_cashflows]
                x2s = [avgs_paths_curr[itm] for itm in itm_cashflows]
                avec0 = self._all_regcoeffs[it_curr]
                assert len(avec0) == 8, 'len(regcoeffs[it_curr]) == 8 is required'
                sum_avec = sum([abs(a) for a in avec0])
                if sum_avec < self._epsilon_lowest:
                    ycnd_expected = [0.0 for _ in itm_cashflows]
                else:
                    design_matrix = NPLeastSquareSimple.design_matrix_cubic_general_model_2vars(basis, x1s, x2s)
                    ycnd_expected = [np.dot(avec0, xr) for xr in design_matrix]
                assert len(itm_cashflows) == len(ycnd_expected), f'{len(itm_cashflows)} == {len(ycnd_expected)} is required'
                exercise_judge = {itm: itm_cashflows[itm] > ycnd_expected[k] for k, itm in enumerate(itm_cashflows)}
                exercise_paths = [itm for itm in exercise_judge if exercise_judge[itm]]
                later_exercise = set(exercise_paths).difference(all_exercise_paths) # before updating all_exercise_paths
                all_exercise_paths.update(exercise_paths)
                cmax_cashflows = self.__max_cashflows_nodiscount_foreward(exercise_paths, curr_cashflows)
                if it_curr > self._min_time_index:
                    self._all_cashflows[it_curr] = [cf if m in later_exercise else 0.0 for m, cf in enumerate(cmax_cashflows)]
                else:
                    self._all_cashflows[it_curr] = cmax_cashflows[:]
        # [last]
        self._all_cashflows[ntimesM1] = [0.0 if m in all_exercise_paths else cf for m, cf in enumerate(self._last_cashflows)]

    def price_american(self) -> float:
        """ american option price based on the optimal early exercise strategy
        """
        assert len(self._all_cashflows) > 0, 'len(self._all_cashflows) > 0 is required'
        idx_min = self._min_time_index
        if self.is_backward:
            time =  idx_min * self.deltaT
            df01 = self.discount(self.rate, time)
            result_cashflows1 = [cf1 for cf1 in self._all_cashflows[idx_min]]
            return df01*sum(result_cashflows1) / self.mpaths
        else:
            result_cashflows0 = []
            for itime, cfs in self._all_cashflows.items():
                term = itime * self.deltaT
                if term >= self.t_lockout:
                    df0T = self.discount(self.rate, term)
                    result_cashflows0.append([df0T*cf for cf in cfs if cf > 0.0])
            result0_flat = [cf0 for row in result_cashflows0 for cf0 in row]
            return sum(result0_flat) / self.mpaths

    def price_european(self) -> float:
        """ european option price discounted back from the maturity
        """
        assert len(self._last_cashflows) > 0, 'len(self._last_cashflows) > 0 is required'
        df0 = self.discount(self.rate, self.t_maturity)
        return sum([df0*payoff for payoff in self._last_cashflows]) / self.mpaths


class DEMOAmericanAsianOptionMonteCarlo:
    def __init__(self) -> None:
        pass

    @classmethod
    def pricing_american_bermudan_asian(cls, is_amc_forward: bool, is_call: bool, iparam1: int, basis: BasisType, order_basis: int, params_monte: tuple):
        parameters_A_S_amer_euro = {
            1: (90.0, 80.0, 0.949, 0.949),
            2: (90.0, 90.0, 3.267, 3.230),
            3: (90.0, 100.0, 7.889, 7.569),
            4: (90.0, 110.0, 14.538, 13.775),
            5: (90.0, 120.0, 22.423, 21.196),
            #
            6: (100.0, 80.0, 1.108, 1.082),
            7: (100.0, 90.0, 3.710, 3.567),
            8: (100.0, 100.0, 8.658, 8.151),
            9: (100.0, 110.0, 15.717, 14.558),
            10: (100.0, 120.0, 23.811, 22.097),
            #
            11: (110.0, 80.0, 1.288, 1.232),
            12: (110.0, 90.0, 4.136, 3.933),
            13: (110.0, 100.0, 9.821, 8.764),
            14: (110.0, 110.0, 17.399, 15.361),
            15: (110.0, 120.0, 25.453, 23.009),
        }
        avg_spot, spot, result_amer_fdm, result_euro_fdm = parameters_A_S_amer_euro[iparam1]
        vol = 0.20
        expiry = 2.0
        strike = 100.0
        rate = 0.06
        qrate = 0.0
        brate = rate - qrate
        t_lockout = 0.25
        avg_delta = 3.0 / 12.0
        df0 = exp(-rate*expiry)
        print(f'# A={avg_spot}, S={spot}, X={strike}, T={expiry}, r={rate}, q={qrate}, b={brate}, vol={vol}, lockout={t_lockout}, avg_period={avg_delta}')
        is_antitheticMC, mc_seed_paths = params_monte
        nsteps1Y = mc_seed_paths['tsteps1Y']
        nsteps = int(nsteps1Y*expiry)
        seed, mpaths = mc_seed_paths['backward']
        print(f'# nsteps:= {nsteps}, mpaths:= {mpaths}')
        dt = expiry / nsteps
        dfi = exp(-rate*dt)
        print(f'# DFi={dfi}, DF0={df0}')
        # [spot]
        price_process = GeneratePriceProcessSimple(seed, nsteps, mpaths, spot, expiry, rate, qrate, vol)
        spot_paths = price_process.spot_paths(is_antitheticMC)
        # [amc.backward]
        amc = OptionAmericanAsianMonteCarlo(avg_spot, spot, strike, expiry, rate, qrate, spot_paths)
        amc.store_all_cashflows_backward(is_call, basis, order_basis, t_lockout, avg_delta)
        # [amc.foreward]
        if is_amc_forward:
            seed, mpaths = mc_seed_paths['foreward']
            price_process.reset_simulation_numbers(seed, nsteps, mpaths)
            spot_paths = price_process.spot_paths(is_antitheticMC)
            amc.reset_spot_paths(spot_paths)
            amc.store_all_cashflows_foreward(is_call, basis, order_basis)
        # amc.show_spot_paths(ndp=1)
        # amc.show_average_spot_paths(ndp=1)
        ndp = 4
        result_amer_amc = round(amc.price_american(), ndp)
        result_euro_amc = round(amc.price_european(), ndp)
        print(f'# american FDM price:= {result_amer_fdm}, AMC price:= {result_amer_amc}')
        print(f'# european FDM price:= {result_euro_fdm}, AMC price:= {result_euro_amc}')
        print('# [END] ----------------------------------------- ')


if __name__ == '__main__':
    msg = '# American Asian option pricing with LSMC'
    print(msg)
    is_call = True
    basis = BasisType.Geometric
    order_basis = 2
    print(basis.name, order_basis)
    is_antitheticMC = True
    mc_seed_paths = {
        'tsteps1Y': (40), # sample test
        'backward': (1932, 100), # sample test
        'foreward': (1932, 100), # sample test
        #
        # 'tsteps1Y': (2**10),
        # 'backward': (1932, 2**12), # org:(1932, 2^14)
        # 'foreward': (7397, 2**14), # org:(7397, 2^14)
    }
    # is_amc_forward = False
    is_amc_forward = True
    params_monte = (is_antitheticMC, mc_seed_paths)
    iparam_sta = 15
    iparam_end = 15
    for iparam1 in range(iparam_sta, iparam_end + 1):
        print(iparam1)
        DEMOAmericanAsianOptionMonteCarlo.pricing_american_bermudan_asian(is_amc_forward, is_call, iparam1, basis, order_basis, params_monte)
