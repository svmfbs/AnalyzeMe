""" American Option Pricing
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
from hepco.model.american_option_quadratic import price_european
from hepco.model.least_square_method import BasisType, NPLeastSquare

sys.dont_write_bytecode = True # dont make .pyc files


class GeneratePriceProcessSimple:
    def __init__(self, seed: int, nsteps: int, mpaths: int, spot: float, expiry: float, rate: float, qrate: float, vol: float) -> None:
        assert spot > 0.0 and expiry > 0.0 and vol > 0.0, 'spot > 0.0 and expiry > 0.0 and vol > 0.0 is required'
        self.spot = spot
        self.dt = expiry / nsteps
        self.mu = rate - qrate - 0.5*vol*vol
        self.volsqrT = vol*np.sqrt(self.dt)
        self.reset_simulation_numbers(seed, nsteps, mpaths)

    def reset_simulation_numbers(self, seed: int, nsteps: int, mpaths: int) -> None:
        assert nsteps > 0 and mpaths > 0, 'nsteps > 0 and mpaths > 0 is required'
        self.nsteps = nsteps
        self.mpaths = mpaths
        rseq = np.random.default_rng(seed)
        self.rand_paths = rseq.standard_normal((nsteps, mpaths))

    def __spot_paths_crude_MC(self, ndp: int=4) -> np.ndarray:
        all_asset_future: np.ndarray = self.mu*self.dt + self.volsqrT*self.rand_paths
        all_asset_values = np.insert(all_asset_future, 0, [0.0 for _ in range(self.mpaths)], axis=0)
        logST = np.log(self.spot) + np.cumsum(all_asset_values, axis=0)
        spot_paths: np.ndarray = np.exp(logST)
        for ir, row in enumerate(spot_paths):
            for jc, v in enumerate(row):
                spot_paths[ir, jc] = round(v, ndp)
        return spot_paths

    def __spot_paths_antithetic_MC(self, ndp: int=4) -> np.ndarray:
        all_asset_future1: np.ndarray = self.mu*self.dt + self.volsqrT*self.rand_paths
        all_asset_future2: np.ndarray = self.mu*self.dt - self.volsqrT*self.rand_paths
        all_asset_future: np.ndarray = np.concatenate([all_asset_future1, all_asset_future2], 1)
        self.mpaths_twice = 2 * self.mpaths
        all_asset_values = np.insert(all_asset_future, 0, [0.0 for _ in range(self.mpaths_twice)], axis=0)
        logST = np.log(self.spot) + np.cumsum(all_asset_values, axis=0)
        spot_paths: np.ndarray = np.exp(logST)
        for ir, row in enumerate(spot_paths):
            for jc, v in enumerate(row):
                spot_paths[ir, jc] = round(v, ndp)
        return spot_paths

    def spot_paths(self, is_antitheticMC: bool, ndp: int=4) -> np.ndarray:
        if is_antitheticMC:
            return self.__spot_paths_antithetic_MC(ndp)
        else:
            return self.__spot_paths_crude_MC(ndp)


class CashflowPayoff:
    def __init__(self, strike: float) -> None:
        self.strike = strike

    def payoff_call(self, spot: float) -> float:
        return max(spot - self.strike, 0.0)

    def payoff_put(self, spot: float) -> float:
        return max(self.strike - spot, 0.0)

    def make_cashflows_at(self, is_call: bool, spot_prices_mpaths: np.ndarray) -> list[float]:
        """ payoff cashflows for all monte paths at interim periods, including the maturity
        cashflows := [w.(Spot(ti, mc) - Strike)]^+
        """
        if is_call:
            return [self.payoff_call(spot) for spot in spot_prices_mpaths]
        else:
            return [self.payoff_put(spot) for spot in spot_prices_mpaths]

    def itm_path_cashflows(self, cashflows_mpaths: list[float]) -> dict[int, float]:
        """ select only in-the-money cashflows and path numbers
        """
        return {jc: cf for jc, cf in enumerate(cashflows_mpaths) if cf > 0.0}


class OptionAmericanMonteCarlo(CashflowPayoff):
    _epsilon = 0.0000000001
    _last_cashflows: list[float]
    _all_cashflows: dict[int, list[float]]
    _all_regcoeffs: dict[int, np.ndarray]

    def __init__(self, spot_price: float, strike: float, t_maturity: float, rate: float, qrate: float, spot_paths: np.ndarray) -> None:
        super().__init__(strike)
        self.spot = spot_price
        # self.strike = strike
        self.t_maturity = t_maturity
        self.rate = rate
        self.qate = qrate
        self.brate = rate - qrate
        self.reset_spot_paths(spot_paths)
        itend = self.ntimesM1
        self._all_regcoeffs = {itend: []}

    def reset_spot_paths(self, spot_paths: np.ndarray) -> None:
        for spot_init in spot_paths[0]:
            assert abs(self.spot - spot_init) < self._epsilon, f'{abs(self.spot - spot_init) < self._epsilon} is required'
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

    def __set_last_payoffs(self, is_call: bool) -> None:
        """ set payoffs at last expiry, put them into all cashflows
        """
        itend = self.ntimesM1
        self._last_cashflows = self.make_cashflows_at(is_call, self.spot_paths[itend, :])
        self._all_cashflows = {itend: self._last_cashflows[:]}

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

    def store_all_cashflows_backward(self, is_call: bool, basis: BasisType, order_basis: int, show_timestep: bool=False) -> None:
        """ calculate all cashflows with optimal early exercise from backward
        """
        print('# AMC backward -----')
        self.is_backward = True
        ntimesM1 = self.ntimesM1
        mpaths = self.mpaths
        rate_c = self.rate
        deltaT = self.deltaT
        self.__set_last_payoffs(is_call)
        for it_curr in reversed(range(1, ntimesM1)):
            it_prev = it_curr + 1
            dfdt = self.discount(rate_c, deltaT)
            spot_paths_curr = self.spot_paths[it_curr, :]
            curr_cashflows = self.make_cashflows_at(is_call, spot_paths_curr)
            itm_cashflows = self.itm_path_cashflows(curr_cashflows)
            xs = [spot_paths_curr[itm] for itm in itm_cashflows]
            ys = [dfdt*self._all_cashflows[it_prev][itm] for itm in itm_cashflows]
            # [regression]
            if len(xs) < order_basis:
                print(f'# [warn]: {it_curr}, len(xs) < order_basis')
                avec0 = np.array([[0.0] for _ in range(order_basis)])
            else:
                avec0 = NPLeastSquare.SolveXinvY(basis, order_basis, np.array(xs), np.array(ys))
            self._all_regcoeffs[it_curr] = avec0
            ycnd_expected = [NPLeastSquare.y_conditional_expectation(basis, order_basis, avec0, x) for x in xs]
            assert len(itm_cashflows) == len(ycnd_expected), f'{len(itm_cashflows)} == {len(ycnd_expected)} is required'
            exercise_judge = {itm: itm_cashflows[itm] > ycnd_expected[k] for k, itm in enumerate(itm_cashflows)}
            exercise_paths = [itm for itm in exercise_judge if exercise_judge[itm]]
            cmax_cashflows = self.__max_cashflows_discounted_backward(it_prev, dfdt, exercise_paths, curr_cashflows)
            self._all_cashflows[it_curr] = cmax_cashflows[:]
            self._all_cashflows[it_prev] = self.zeros[:]
            if show_timestep:
                print(it_curr)

    def store_all_cashflows_foreward(self, is_call: bool, basis: BasisType, order_basis: int, show_timestep: bool=False) -> None:
        """ calculate all cashflows from foreward with regressions
        """
        print('# AMC foreward -----')
        self.is_backward = False
        ntimesM1 = self.ntimesM1
        mpaths = self.mpaths
        rate_c = self.rate
        deltaT = self.deltaT
        all_exercise_paths = set()
        self.__set_last_payoffs(is_call)
        for it_curr in range(1, ntimesM1):
            # dfdt = self.discount(rate_c, deltaT)
            spot_paths_curr = self.spot_paths[it_curr, :]
            curr_cashflows = self.make_cashflows_at(is_call, spot_paths_curr)
            itm_cashflows = self.itm_path_cashflows(curr_cashflows)
            xs = [spot_paths_curr[itm] for itm in itm_cashflows]
            avec0 = self._all_regcoeffs[it_curr]
            assert len(avec0) == order_basis, 'len(regcoeffs[it_curr]) == order_basis is required'
            ycnd_expected = [NPLeastSquare.y_conditional_expectation(basis, order_basis, avec0, x) for x in xs]
            assert len(itm_cashflows) == len(ycnd_expected), f'{len(itm_cashflows)} == {len(ycnd_expected)} is required'
            exercise_judge = {itm: itm_cashflows[itm] > ycnd_expected[k] for k, itm in enumerate(itm_cashflows)}
            exercise_paths = [itm for itm in exercise_judge if exercise_judge[itm]]
            later_exercise = set(exercise_paths).difference(all_exercise_paths) # before updating all_exercise_paths
            all_exercise_paths.update(exercise_paths)
            cmax_cashflows = self.__max_cashflows_nodiscount_foreward(exercise_paths, curr_cashflows)
            if it_curr > 1:
                self._all_cashflows[it_curr] = [cf if m in later_exercise else 0.0 for m, cf in enumerate(cmax_cashflows)]
            else:
                self._all_cashflows[it_curr] = cmax_cashflows[:]
            if show_timestep:
                print(it_curr)
        # [last]
        self._all_cashflows[ntimesM1] = [0.0 if m in all_exercise_paths else cf for m, cf in enumerate(self._last_cashflows)]

    def price_american(self) -> float:
        """ american option price based on the optimal early exercise strategy
        """
        assert len(self._all_cashflows) > 0, 'len(self._all_cashflows) > 0 is required'
        if self.is_backward:
            df01 = self.discount(self.rate, self.deltaT)
            result_cashflows1 = [cf1 for cf1 in self._all_cashflows[1]]
            return df01*sum(result_cashflows1)/self.mpaths
        else:
            result_cashflows0 = []
            for itime, cfs in self._all_cashflows.items():
                term = itime * self.deltaT
                df0T = self.discount(self.rate, term)
                result_cashflows0.append([df0T*cf for cf in cfs if cf > 0.0])
            result0_flat = [cf0 for row in result_cashflows0 for cf0 in row]
            return sum(result0_flat)/self.mpaths

    def price_european(self) -> float:
        """ european option price discounted back from the maturity
        """
        assert len(self._last_cashflows) > 0, 'len(self._last_cashflows) > 0 is required'
        df0 = self.discount(self.rate, self.t_maturity)
        return sum([df0*payoff for payoff in self._last_cashflows])/self.mpaths


class DEMOAmericanOptionMonteCarlo:
    def __init__(self) -> None:
        pass

    @classmethod
    def pricing_american_sample(cls, is_call: bool, iparam1: int, basis: BasisType, order_basis: int, params_monte: tuple):
        parameters_s_v_t_result = {
            1: (36.0, 0.20, 1.0, 4.478),
            2: (36.0, 0.20, 2.0, 4.840),
            3: (36.0, 0.40, 1.0, 7.101),
            4: (36.0, 0.40, 2.0, 8.508),
            #
            5: (38.0, 0.20, 1.0, 3.250),
            6: (38.0, 0.20, 2.0, 3.745),
            7: (38.0, 0.40, 1.0, 6.148),
            8: (38.0, 0.40, 2.0, 7.670),
            #
            9: (40.0, 0.20, 1.0, 2.314),
            10: (40.0, 0.20, 2.0, 2.885),
            11: (40.0, 0.40, 1.0, 5.312),
            12: (40.0, 0.40, 2.0, 6.920),
            #
            13: (42.0, 0.20, 1.0, 1.617),
            14: (42.0, 0.20, 2.0, 2.212),
            15: (42.0, 0.40, 1.0, 4.582),
            16: (42.0, 0.40, 2.0, 6.248),
            #
            17: (44.0, 0.20, 1.0, 1.110),
            18: (44.0, 0.20, 2.0, 1.690),
            19: (44.0, 0.40, 1.0, 3.948),
            20: (44.0, 0.40, 2.0, 5.647),
        }
        spot, vol, expiry, result_fdm = parameters_s_v_t_result[iparam1]
        strike = 40.0
        rate = 0.06
        qrate = 0.0
        brate = rate - qrate
        df0 = exp(-rate*expiry)
        print(f'# S={spot}, X={strike}, T={expiry}, r={rate}, q={qrate}, b={brate}, vol={vol}')
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
        # [reg.basis]
        # basis = BasisType.Geometric
        # order_basis = 3
        # [amc.backward]
        amc = OptionAmericanMonteCarlo(spot, strike, expiry, rate, qrate, spot_paths)
        # amc.show_spot_paths(ndp=4)
        amc.store_all_cashflows_backward(is_call, basis, order_basis)
        # amc.show_regcoeffs(ndp=4)
        # [amc.foreward]
        seed, mpaths = mc_seed_paths['foreward']
        price_process.reset_simulation_numbers(seed, nsteps, mpaths)
        spot_paths = price_process.spot_paths(is_antitheticMC)
        amc.reset_spot_paths(spot_paths)
        amc.store_all_cashflows_foreward(is_call, basis, order_basis)
        # amc.show_cashflows(ndp=4)
        print('# [results] -----')
        print(f'# is call option:= {"call" if is_call else "put"}')
        ndp = 5
        price_amer = round(amc.price_american(), ndp)
        price_euro = round(amc.price_european(), ndp)
        result_bsm = round(price_european(is_call, df0, spot, strike, rate, qrate, vol, expiry), ndp)
        print(f'# american price:= {price_amer}, FDM price:= {result_fdm}, diff:= {abs(price_amer - result_fdm)}')
        print(f'# european price:= {price_euro}, BSM price:= {result_bsm}, diff:= {abs(price_euro - result_bsm)}')
        print('# [END] ----------------------------------------- ')
        if iparam1 == 1:
            assert price_amer == 4.47178, f'{price_amer} == 4.47178 is required'


if __name__ == '__main__':
    msg = '# American option pricing with LSMC'
    print(msg)
    is_call = False
    basis = BasisType.Geometric
    order_basis = 5
    print(basis.name, order_basis)
    is_antitheticMC = True
    mc_seed_paths = {
        # 'tsteps1Y': (5), # sample test
        # 'backward': (1932, 10), # sample test
        'tsteps1Y': (2**6),
        'backward': (1932, 2**11), # org
        'foreward': (7397, 2**16),
    }
    params_monte = (is_antitheticMC, mc_seed_paths)
    DEMOAmericanOptionMonteCarlo.pricing_american_sample(is_call, 1, basis, order_basis, params_monte)
    # sys.exit()
    # for iparam1 in range(1, 21):
    #     print(f'# [{iparam1}] #########################################')
    #     DEMOAmericanOptionMonteCarlo.pricing_american_sample(is_call, iparam1, basis, order_basis)
