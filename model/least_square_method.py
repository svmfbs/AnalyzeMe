""" American Option Pricing
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from enum import auto, Enum
from math import exp
from scipy.linalg import solve
import numpy as np


sys.dont_write_bytecode = True # dont make .pyc files


class BasisType(Enum):
    Geometric = auto()
    Laguerre = auto()
    Hermite = auto()
    Legendre = auto()
    Chebyshev1st = auto()


class PolynomialOrder10:
    def __init__(self) -> None:
        pass

    @classmethod
    def __geometric(cls, morder: int, x: float) -> float:
        if morder == 0:
            return 1.0
        return np.float_power(x, morder)

    @classmethod
    def __laguerre(cls, morder: int, x: float) -> float:
        factor = exp(-0.5*x)
        # factor = 1.0
        if morder == 0:
            return factor
        elif morder == 1:
            return factor*(1.0 - x)
        elif morder == 2:
            x2 = x*x
            return factor*(2.0 - 4.0*x + x2)/2.0
        elif morder == 3:
            x2 = x*x
            x3 = x2*x
            return factor*(6.0 - 18.0*x + 9.0*x2 - x3)/6.0
        elif morder == 4:
            x2 = x*x
            x3 = x2*x
            x4 = x3*x
            return factor*(24.0 - 96.0*x + 72.0*x2 - 16.0*x3 + x4)/24.0
        elif morder == 5:
            x2 = x*x
            x3 = x2*x
            x4 = x3*x
            x5 = x4*x
            return factor*(120.0 - 600.0*x + 600.0*x2 - 200*x3 + 25.0*x4 - x5)/120.0
        elif morder == 6:
            x2 = x*x
            x3 = x2*x
            x4 = x3*x
            x5 = x4*x
            x6 = x5*x
            return factor*(720.0 - 4320.0*x + 5400.0*x2 - 2400*x3 + 450.0*x4 - 36.0*x5 + x6)/720.0
        elif morder == 7:
            x2 = x*x
            x3 = x2*x
            x4 = x3*x
            x5 = x4*x
            x6 = x5*x
            x7 = x6*x
            return factor*(5040.0 - 35280.0*x + 52920.0*x2 - 29400*x3 + 7350.0*x4 - 882.0*x5 + 49.0*x6 - x7)/5040.0
        elif morder == 8:
            x2 = x*x
            x3 = x2*x
            x4 = x3*x
            x5 = x4*x
            x6 = x5*x
            x7 = x6*x
            x8 = x7*x
            return factor*(40320.0 - 322560.0*x + 564480.0*x2 - 376320*x3 + 117600.0*x4 - 18816.0*x5 + 1568.0*x6 - 64.0*x7 + x8)/40320.0
        elif morder == 9:
            x2 = x*x
            x3 = x2*x
            x4 = x3*x
            x5 = x4*x
            x6 = x5*x
            x7 = x6*x
            x8 = x7*x
            x9 = x8*x
            return factor*(362880.0 - 3265920.0*x + 6531840.0*x2 - 5080320*x3 + 1905120.0*x4 - 381024.0*x5 + 42366.0*x6 - 2592.0*x7 + 81.0*x8 - x9)/362880.0
        elif morder == 10:
            x2 = x*x
            x3 = x2*x
            x4 = x3*x
            x5 = x4*x
            x6 = x5*x
            x7 = x6*x
            x8 = x7*x
            x9 = x8*x
            x10 = x9*x
            return factor*(3628800.0 - 36288000.0*x + 81648000.0*x2 - 72576000.0*x3 + 31752000.0*x4 - 7620480.0*x5 + 1058400.0*x6 - 86400.0*x7 + 4050.0*x8 - 100.0*x9 + x10)/3628800.0
        else:
            raise ValueError('# error: not suitable order')

    @classmethod
    def __hermite(cls, morder: int, x: float) -> float:
        if morder == 0:
            return 1.0
        elif morder == 1:
            return 2.0*x
        elif morder == 2:
            x2 = x*x
            return 4.0*x2 - 2.0
        elif morder == 3:
            x2 = x*x
            x3 = x2*x
            return 8.0*x3 - 12.0*x
        elif morder == 4:
            x2 = x*x
            x4 = x2*x2
            return 16.0*x4 - 48.0*x2 + 12.0
        elif morder == 5:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            return 32.0*x5 - 160.0*x3 + 120.0*x
        elif morder == 6:
            x2 = x*x
            x3 = x2*x
            x4 = x2*x2
            x6 = x3*x3
            return 64.0*x6 - 480.0*x4 + 720.0*x2 - 120.0
        elif morder == 7:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            x7 = x5*x2
            return 128.0*x7 - 1344.0*x5 + 3360.0*x3 - 1680.0*x
        elif morder == 8:
            x2 = x*x
            x3 = x2*x
            x4 = x2*x2
            x6 = x3*x3
            x8 = x4*x4
            return 256.0*x8 - 3584.0*x6 + 13440.0*x4 - 13440.0*x2 + 1680.0
        elif morder == 9:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            x7 = x5*x2
            x9 = x7*x2
            return 512.0*x9 - 9216.0*x7 + 48384.0*x5 - 80640.0*x3 + 302400.0*x
        elif morder == 10:
            x2 = x*x
            x3 = x2*x
            x4 = x2*x2
            x6 = x3*x3
            x8 = x4*x4
            x10 = x8*x2
            return 1024.0*x10 - 23040.0*x8 + 161280.0*x6 - 403200.0*x4 + 302400.0*x2 - 30240.0
        else:
            raise ValueError('# error: not suitable order')

    @classmethod
    def __legendre(cls, morder: int, x: float) -> float:
        if morder == 0:
            return 1.0
        elif morder == 1:
            return x
        elif morder == 2:
            x2 = x*x
            return (3.0*x2 - 1.0)/2.0
        elif morder == 3:
            x2 = x*x
            x3 = x2*x
            return (5.0*x3 - 3.0*x)/2.0
        elif morder == 4:
            x2 = x*x
            x4 = x2*x2
            return (35.0*x4 - 30.0*x2 + 3.0)/8.0
        elif morder == 5:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            return (63.0*x5 - 70.0*x3 + 15.0*x)/8.0
        elif morder == 6:
            x2 = x*x
            x3 = x2*x
            x4 = x2*x2
            x6 = x3*x3
            return (231.0*x6 - 315.0*x4 + 105.0*x2 - 5.0)/16.0
        elif morder == 7:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            x7 = x5*x2
            return (429.0*x7 - 693.0*x5 + 315.0*x3 - 35.0*x)/16.0
        elif morder == 8:
            x2 = x*x
            x3 = x2*x
            x4 = x2*x2
            x6 = x3*x3
            x8 = x4*x4
            return (6435.0*x8 - 12012.0*x6 + 6930.0*x4 - 1260.0*x2 + 35.0)/128.0
        elif morder == 9:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            x7 = x5*x2
            x9 = x7*x2
            return (12155.0*x9 - 25740.0*x7 + 18018.0*x5 - 4620.0*x3 + 315.0*x)/128.0
        elif morder == 10:
            x2 = x*x
            x3 = x2*x
            x4 = x2*x2
            x6 = x3*x3
            x8 = x4*x4
            x10 = x8*x2
            return (46189.0*x10 - 109395.0*x8 + 90090.0*x6 - 300300.0*x4 + 3465.0*x2 - 63.0)/256.0
        else:
            raise ValueError('# error: not suitable order')

    @classmethod
    def __chebyshev1st(cls, morder: int, x: float) -> float:
        if morder == 0:
            return 1.0
        elif morder == 1:
            return x
        elif morder == 2:
            x2 = x*x
            return 2.0*x2 - 1.0
        elif morder == 3:
            x3 = x*x*x
            return 4.0*x3 - 3.0*x
        elif morder == 4:
            x2 = x*x
            x4 = x2*x2
            return 8.0*x4 - 8.0*x2 + 1.0
        elif morder == 5:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            return 16.0*x5 -20.0*x3 + 5.0*x
        elif morder == 6:
            x2 = x*x
            x4 = x2*x2
            x6 = x4*x2
            return 32.0*x6 - 48.0*x4 + 18.0*x2 - 1.0
        elif morder == 7:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            x7 = x5*x2
            return 64.0*x7 - 112.0*x5 + 56.0*x3 - 7.0*x
        elif morder == 8:
            x2 = x*x
            x4 = x2*x2
            x6 = x4*x2
            x8 = x6*x2
            return 128.0*x8 - 256.0*x6 + 160.0*x4 - 32.0*x2 + 1.0
        elif morder == 9:
            x2 = x*x
            x3 = x2*x
            x5 = x3*x2
            x7 = x5*x2
            x9 = x7*x2
            return 259.0*x9 - 576.0*x7 + 432.0*x5 - 120.0*x3 + 9.0*x
        elif morder == 10:
            x2 = x*x
            x4 = x2*x2
            x6 = x4*x2
            x8 = x6*x2
            x10 = x8*x2
            return 512.0*x10 - 1280.0*x8 + 1120.0*x6 - 400.0*x4 + 50.0*x2 - 1.0
        else:
            raise ValueError('# error: not suitable order')

    @classmethod
    def func(cls, basis: BasisType, morder: int, x: float) -> float:
        assert 0 <= morder <= 10, f'1 <= {morder} <= 10 is required'
        """ polynomial basis function
        """
        if basis == BasisType.Geometric:
            return cls.__geometric(morder, x)
        elif basis == BasisType.Laguerre:
            return cls.__laguerre(morder, x)
        elif basis == BasisType.Hermite:
            return cls.__hermite(morder, x)
        elif basis == BasisType.Legendre:
            return cls.__legendre(morder, x)
        elif basis == BasisType.Chebyshev1st:
            return cls.__chebyshev1st(morder, x)
        else:
            raise ValueError('# error: basis type is wrong.')


class NPLeastSquareSimple:
    _epsilon = 1.0e-128

    def __init__(self) -> None:
        pass

    @classmethod
    def MatrixX(cls, basis: BasisType, morder: int, xs: np.ndarray) -> np.ndarray:
        """ y = a0 + a1.x1 + a2.x2 + a3.x3 + a4.x4 + a5.x5 + ... + aM.xM
        observed data points:= N (M<=N)
        m00 = N
        m01 = m10 = sum_1^N(x_i)
        m02 = m20 = sum_1^N(x_i^2)
        m03 = m30 = sum_1^N(x_i^3)
        m04 = m40 = sum_1^N(x_i^4)
        m14 = m41 = sum_1^N(x_i^5)
        m24 = m42 = sum_1^N(x_i^6)
        m34 = m43 = sum_1^N(x_i^7)
        m11 = m02
        m22 = m04 = m13
        m33 = m24
        m44 = sum_1^N(x_i^8)
        [ m00, m01, m02, m03, m04 ]
        [ m10, m11, m12, m13, m14 ]
        [ m20, m21, m22, m23, m24 ]
        [ m30, m31, m32, m33, m34 ]
        [ m40, m41, m42, m43, m44 ]
        """
        nmax = len(xs)
        assert nmax > 0, f'{nmax} > 0 is required'
        assert 0 < morder <= nmax, f'0 < {morder} <= {nmax} is required'
        xs_pow_n = [np.array([PolynomialOrder10.func(basis, p, x) for x in xs]) for p in range(morder)]
        # xhat = np.matrix(xs_pow_n)
        # return xhat*xhat.T
        xmat = np.zeros((morder, morder))
        for ir in range(morder):
            xmat[ir, ir] = np.dot(xs_pow_n[ir], xs_pow_n[ir])
            for jc in range(ir):
                xmat[ir, jc] = np.dot(xs_pow_n[jc], xs_pow_n[ir])
                xmat[jc, ir] = xmat[ir, jc]
        return xmat

    @classmethod
    def VectorXY(cls, basis: BasisType, morder: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """ y = a0 + a1.x1 + a2.x2 + a3.x3 + a4.x4 + a5.x5 + ... + aM.xM
        observed data points:= N (M<=N)
        xy00 = sum_1^N(y_i)
        xy10 = sum_1^N(x_i.y_i)
        xy20 = sum_1^N(x_i^2.y_i)
        ...
        xyM0 = sum_1^N(x_i^M.y_i)
        """
        nmax = len(xs)
        assert nmax == len(ys), f'{nmax} == {len(ys)} is required'
        assert 0 < morder <= nmax, f'0 < {morder} <= {nmax} is required'
        xs_pow_n = [np.array([PolynomialOrder10.func(basis, p, x) for x in xs]) for p in range(morder)]
        yvec = np.zeros((morder, 1))
        for ir in range(morder):
            yvec[ir, 0] = np.dot(xs_pow_n[ir], ys)
        # yvec = [np.dot(xs_pow_n[p], ys) for p in range(morder)]
        return yvec

    @classmethod
    def SolveXinvY(cls, basis: BasisType, morder: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """ Solve Xmat.Avec = Yvec (X.a = Y) in terms of Avec
        Avec = Xmat^{-1}.Yvec (a = X^-1.Y)
        """
        xmat = cls.MatrixX(basis, morder, xs)
        xdet = np.linalg.det(xmat)
        if abs(xdet) > cls._epsilon:
            yvec = cls.VectorXY(basis, morder, xs, ys)
            return solve(xmat, yvec)
        else:
            raise ValueError(f'# error: determinant of X is zero: {xdet}')

    @classmethod
    def y_conditional_expectation(cls, basis: BasisType, morder: int, avec0: np.ndarray, x: float) -> float:
        """ y = E[y|x] = a0 + a1.x^1 + a2.x^2 + a3.x^3 + ... + aM.x^M
            = (a0, a1, a2, a3, ,,, aM).(1, x^1, x^2, x^3, ,,, x^M)
        """
        assert len(avec0) == morder, 'len(avec0) == morder is required'
        assert avec0.shape[1] == 1, 'avec0.shape[1] == 1 is required'
        return sum([avec0[p, 0]*PolynomialOrder10.func(basis, p, x) for p in range(morder)])

    @classmethod
    def __make_design_matrix(cls, dict_design_array_t: dict[int, list[float]]) -> np.ndarray:
        design_matrix_t = np.array([dict_design_array_t[i] for i in dict_design_array_t.keys()])
        design_matrix = design_matrix_t.T
        return design_matrix

    @classmethod
    def design_matrix_quadratic_model_2vars(cls, x1s: list[float], x2s: list[float]) -> np.ndarray:
        """ design matrix of a quadratic model in two variables
        design matrix: X = [1, x1, x2, x1^2, x2^2, x1.x2]
        """
        assert len(x1s) == len(x2s), 'len(x1s) == len(x2s) is required'
        dict_design_array_t = {
            0: [1.0 for _ in x1s],
            1: x1s[:],
            2: x2s[:],
            3: [x1*x1 for x1 in x1s],
            4: [x2*x2 for x2 in x2s],
            5: [x1*x2 for x1, x2 in zip(x1s, x2s)],
        }
        return cls.__make_design_matrix(dict_design_array_t)

    @classmethod
    def design_matrix_cubic_model_2vars(cls, x1s: list[float], x2s: list[float]) -> np.ndarray:
        """ design matrix of a cubic model in two variables
        design matrix: X = [1, x1, x2, x1^2, x2^2, x1.x2, x1.x2^2, x1^2.x2]
        """
        assert len(x1s) == len(x2s), 'len(x1s) == len(x2s) is required'
        dict_design_array_t = {
            0: [1.0 for _ in x1s],
            1: x1s[:],
            2: x2s[:],
            3: [x1*x1 for x1 in x1s],
            4: [x2*x2 for x2 in x2s],
            5: [x1*x2 for x1, x2 in zip(x1s, x2s)],
            6: [x1*x2*x2 for x1, x2 in zip(x1s, x2s)],
            7: [x1*x1*x2 for x1, x2 in zip(x1s, x2s)],
        }
        return cls.__make_design_matrix(dict_design_array_t)

    @classmethod
    def design_matrix_cubic_general_model_2vars(cls, basis: BasisType, x1s: list[float], x2s: list[float]) -> np.ndarray:
        """ design matrix of a cubic model in two variables
        design matrix: X = [1, x1, x2, x1^2, x2^2, x1.x2, x1.x2^2, x1^2.x2]
        """
        assert len(x1s) == len(x2s), 'len(x1s) == len(x2s) is required'
        n1, n2 = 1, 2
        if BasisType.Laguerre == basis:
            n1, n2 = 0, 1
        dict_design_array_t = {
            0: [1.0 for _ in x1s],
            1: [PolynomialOrder10.func(basis, n1, x1) for x1 in x1s],
            2: [PolynomialOrder10.func(basis, n1, x2) for x2 in x2s],
            3: [PolynomialOrder10.func(basis, n2, x1) for x1 in x1s],
            4: [PolynomialOrder10.func(basis, n2, x2) for x2 in x2s],
            5: [PolynomialOrder10.func(basis, n1, x1)*PolynomialOrder10.func(basis, n1, x2) for x1, x2 in zip(x1s, x2s)],
            6: [PolynomialOrder10.func(basis, n1, x1)*PolynomialOrder10.func(basis, n2, x2) for x1, x2 in zip(x1s, x2s)],
            7: [PolynomialOrder10.func(basis, n2, x1)*PolynomialOrder10.func(basis, n1, x2) for x1, x2 in zip(x1s, x2s)],
        }
        return cls.__make_design_matrix(dict_design_array_t)

    @classmethod
    def beta_coefficients(cls, design_matrix: np.ndarray, ys_vec: np.ndarray) -> np.ndarray:
        """ y := beta0 + beta1.x1 + beta2.x2 + beta3.x3 + beta4.x4 + beta5.x5 + epsilon
        beta = [beta0, beta1, beta2, beta3, beta4, beta5]
        beta_hat = (X^t.X)^-1.X^t.y
        """
        XtX = design_matrix.T @ design_matrix
        XtY = design_matrix.T @ ys_vec
        detXtX = np.linalg.det(XtX)
        if abs(detXtX) > cls._epsilon:
            XtX_inv = np.linalg.inv(XtX)
        else:
            try:
                XtX_inv = np.linalg.pinv(XtX)
            except:
                # [tentative]
                XtX_inv = np.zeros_like(XtX)
        beta_hat = XtX_inv @ XtY
        return beta_hat


class DEMOLeastSquareMethod:
    def __init__(self) -> None:
        pass

    @classmethod
    def basis_polynomials(cls):
        basis = BasisType.Hermite
        xs = np.linspace(0, 10, 100)
        for x in xs:
            gL0 = PolynomialOrder10.func(basis, 0, x)
            gL1 = PolynomialOrder10.func(basis, 1, x)
            gL2 = PolynomialOrder10.func(basis, 2, x)
            gL3 = PolynomialOrder10.func(basis, 3, x)
            gL4 = PolynomialOrder10.func(basis, 4, x)
            gL5 = PolynomialOrder10.func(basis, 5, x)
            print(x, gL0, gL1, gL2, gL3, gL4, gL5)

    @classmethod
    def least_square_method(cls):
        # xs = [1.0816, 1.0628, 1.0754, 0.9316, 1.0402, 0.9872, 0.8267]
        # ys = [0.1169, 0.1037, 0.125, 0.101, 0.1411, 0.1396, 0.2621]
        xs = [0.8837, 1.0273, 0.9506, 0.9185, 1.0247, 1.0426, 1.0473, 0.8753, 0.934]
        ys = [0.0000, 0.0000, 0.0000, 0.1659, 0.0000, 0.0000, 0.0000, 0.2693, 0.000]
        basis = BasisType.Geometric
        order_basis = 3
        avec0 = NPLeastSquareSimple.SolveXinvY(basis, order_basis, np.array(xs), np.array(ys))
        ycnd_expected = [NPLeastSquareSimple.y_conditional_expectation(basis, order_basis, avec0, x) for x in xs]
        for i, yc in enumerate(ycnd_expected):
            print(i, round(yc,6), xs[i], ys[i])

    @classmethod
    def quadratic_model_in_two_variables(cls):
        ts = [200.0, 250.0, 200.0, 250.0, 189.65, 260.35, 225.0, 225.0, 225.0, 225.0, 225.0, 225.0]
        cs = [15.0, 15.0, 25.0, 25.0, 20.0, 20.0, 12.93, 27.07, 20.0, 20.0, 20.0, 20.0]
        ys = [43.0, 78.0, 69.0, 73.0, 48.0, 76.0, 65.0, 74.0, 76.0, 79.0, 83.0, 81.0]
        x1s = [(t - 225.0)/25.0 for t in ts]
        x2s = [(c - 20.0)/5.0 for c in cs]
        design_matrix = NPLeastSquareSimple.design_matrix_quadratic_model_2vars(x1s, x2s)
        # design_matrix = NPLeastSquareSimple.design_matrix_cubic_model_2vars(x1s, x2s)
        beta_coeffs = NPLeastSquareSimple.beta_coefficients(design_matrix, np.array(ys))
        for i, xr in enumerate(design_matrix):
            yhat = np.dot(beta_coeffs, xr)
            diff = yhat - ys[i]
            assert abs(diff) < 3.76, 'abs(diff) < 3.76 is required'
            print(f'{i}, {yhat:.4f}, {ys[i]:.4f}, {diff:.4f}')


if __name__ == '__main__':
    msg = '# Least Square Method'
    print(msg)
    # DEMOLeastSquareMethod.basis_polynomials()
    # DEMOLeastSquareMethod.least_square_method()
    DEMOLeastSquareMethod.quadratic_model_in_two_variables()
