""" This is Math utility functions program """
import math
import statistics
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random as nr

sys.dont_write_bytecode = True # dont make .pyc files

def main_random():
    print('# set random seed')
    nr.seed(5)
    urand_mat = nr.rand(5, 5)
    nrand_mat = nr.randn(5, 5)
    print(f'# uniform random      := \n{urand_mat}')
    print(f'# normal gauss random := \n{nrand_mat}')
    mean = 50
    sigma = 10
    grand = nr.normal(mean, sigma)
    print(f'# gauss dist with : ({mean}, {sigma}) = {grand}')
    # plt.hist(nr.randn(10000), bins=100)
    # plt.show()
    # uniform random dice ###########
    dice = list(range(1, 7))
    for size in range(10):
        samples = np.random.choice(dice, size)
        # print(samples)
    dice_mat = np.random.choice(dice, (2,3))
    print(dice_mat)
    # non-uniform random dice ###########
    prob = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]
    # prob = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
    samples = np.random.choice(a=dice, size=100000, p=prob)
    tmp = Counter(samples)
    X = sorted(tmp.keys())
    Y = [tmp[x] for x in X]
    print(Y)
    # plt.bar(X, Y)
    # plt.show()

class MathBasic():
    @classmethod
    def fibo_seq(cls, n):
        """ Fibonatti sequence """
        result = []
        a = 1.0
        b = 1.0
        while b < n:
            result.append(b)
            val = a + b
            b = a
            a = val
        return result

    @classmethod
    def hypertan(cls, x, xlower=-20.0, xupper=20.0):
        """ Hyper tangent function, x in (-20, 20) """
        if x < xlower:
            return -1.0
        elif x > xupper:
            return 1.0
        else:
            return math.tanh(x)

    @classmethod
    def softmax(cls, oSums):
        """ Soft-max function """
        num = len(oSums)
        m = max(oSums)
        divisor = 0.0
        for k in range(num):
            divisor += math.exp(oSums[k]-m)
        if divisor == 0.0:
            print('Error: divisor = 0')
        result = np.zeros(shape=[num], dtype=np.float32)
        for k in range(len(result)):
            result[k] = math.exp(oSums[k]-m)/divisor
        return result


class Statistics():
    @classmethod
    def range(cls, seq):
        return (min(seq), max(seq))
    
    @classmethod
    def mean(cls, seq):
        return sum(seq) / len(seq)

    @classmethod
    def variance(cls, seq):
        mean = cls.mean(seq)
        total = sum(map(lambda x:(x-mean)**2, seq))
        return total / len(seq)
    
    @classmethod
    def stdev(cls, seq):
        return cls.variance(seq)**0.5

    @classmethod
    def median(cls, seq):
        s = sorted(seq)
        num = len(s)
        is_even = num % 2 == 0
        intdiv = num // 2
        midval = s[intdiv]
        return (midval + s[intdiv - 1]) / 2 if is_even else midval

    @classmethod
    def mode(cls, seq):
        val_cnt = Counter(seq).most_common()
        max_cnt = val_cnt[0][1]
        modes_max_only = filter(lambda x:x[1] == max_cnt, val_cnt)
        return [valcnt[0] for valcnt in modes_max_only]

    @classmethod
    def corr(cls, xs, ys):
        if len(xs) != len(ys):
            return "different length"
        sumx1 = sum(xs)
        sumy1 = sum(ys)
        sumx2 = sum(map(lambda x:x*x, xs))
        sumy2 = sum(map(lambda y:y*y, ys))
        sumxy = sum(map((lambda x, y:x*y), xs, ys))
        nsize = len(xs)
        try:
            ncovr = (nsize * sumxy - sumx1 * sumy1)
            nvarx = (nsize * sumx2 - sumx1 * sumx1)
            nvary = (nsize * sumy2 - sumy1 * sumy1)
            return ncovr / math.sqrt(nvarx * nvary)
        except:
            return 0.0

    @classmethod
    def stat(cls, seq, nround=2):
        if len(seq) == 0:
            print("No-element")
        else:
            print("要素数  :", len(seq))
            print("平均    :", round(cls.mean(seq), nround))
            print("範囲    :", cls.range(seq))
            print("最頻値  :", *cls.mode(seq))
            print("中央値  :", cls.median(seq))
            print("分散    :", round(cls.variance(seq), nround))
            print("標準偏差:", round(cls.stdev(seq), nround))
            xs = [1, 2, 3]
            ys = [5, 5, 6]
            print("相関係数:", cls.corr(xs, ys))


class MatrixUtils():
    @classmethod
    def inner_product(cls, matA, matB, matC=None):
        ''' Multiply matrix, A.B or A.B.C '''
        if matC is None:
            # A.B
            if matA.shape[1] == matB.shape[0]:
                return np.dot(matA, matB)
        else:
            # A.B.C
            matBC = cls.inner_product(matB, matC)
            if matA.shape[1] == matBC.shape[0]:
                return np.dot(matA, matBC)
        print(f'# Error: in {sys._getframe().f_code.co_name}')
        return None

    @classmethod
    def extract_submatrix_1dim(cls, matA, k_use, idx_del_axis):
        ''' Reduce matrix, ([0:M],[0:N]) => ([0:M],[0:k]) or ([0:k],[0:N]) '''
        length_axis = matA.shape[idx_del_axis]
        delete_axis = range(k_use, length_axis)
        return np.delete(matA, delete_axis, axis=idx_del_axis)

    @classmethod
    def extract_submatrix_2dim(cls, matA, k_use, idx_del_row=0, idx_del_col=1):
        ''' Reduce matrix, ([0:M],[0:N]) => ([0:M],[0:k]) or ([0:k],[0:k]) '''
        matA_del_row = cls.extract_submatrix_1dim(matA, k_use, idx_del_row)
        return cls.extract_submatrix_1dim(matA_del_row, k_use, idx_del_col)


class SVD():
    ''' Sp = U.S.V^t : [MxN] = [MxN].[NxN].[NxN] '''
    _IdxRow = 0
    _IdxCol = 1

    def __init__(self, mat):
        self.mat_S = mat

    def execute(self, isfullSVD=False):
        (U, s, V) = np.linalg.svd(self.mat_S, full_matrices=isfullSVD)
        self.Sigma = np.diag(s)
        self.mat_U = U
        self.mat_V = V.T
    
    def calc_USVt(self):
        self.mat_Sp = MatrixUtils.inner_product(self.mat_U, self.Sigma, self.mat_V.T)
        return self.mat_Sp

    def allclose(self, rel_tol=1e-05, abs_tol=1e-08):
        try:
            return np.allclose(self.mat_S, self.mat_Sp, rel_tol, abs_tol)
        except:
            print("# This variable is not defined, First calc_USVt()")

    def isclose(self, rel_tol=1e-05, abs_tol=1e-08):
        try:
            return np.isclose(self.mat_S, self.mat_Sp, rel_tol, abs_tol)
        except:
            print("# This variable is not defined, First calc_USVt()")

    def array_equal(self, rel_tol=1e-05, abs_tol=1e-08):
        try:
            return np.array_equal(self.mat_S, self.mat_Sp)
        except:
            print("# This variable is not defined, First calc_USVt()")

    def reduce_USVt(self, k_use):
        mat_Uk = MatrixUtils.extract_submatrix_1dim(self.mat_U, k_use, self._IdxCol)
        sigmak = MatrixUtils.extract_submatrix_2dim(self.Sigma, k_use, self._IdxRow, self._IdxCol)
        mat_Vk = MatrixUtils.extract_submatrix_1dim(self.mat_V, k_use, self._IdxCol)
        return MatrixUtils.inner_product(mat_Uk, sigmak, mat_Vk.T)

    def check_matrix(self, actual_mat, expect_mat, abs_tol=1.0e-14):
        for ir in range(expect_mat.shape[0]):
            for jc in range(expect_mat.shape[1]):
                assert abs(actual_mat[ir][jc] - expect_mat[ir][jc]) < abs_tol


def test_statistics():
    print(f'# TEST: in {sys._getframe().f_code.co_name}')
    nround = 2
    seq = [100, 60, 50, 70, 100]
    assert 5 == len(seq)
    assert 76.0 == round(Statistics.mean(seq), nround)
    assert (50, 100) == Statistics.range(seq)
    assert 100.0 == Statistics.mode(seq)[0]
    assert 70.0 == Statistics.median(seq)
    assert 424.0 == round(Statistics.variance(seq), nround)
    assert 20.59 == round(Statistics.stdev(seq), nround)
    xs = [1, 2, 3]
    ys = [5, 5, 6]
    assert 0.8660254037844387 == Statistics.corr(xs, ys)


def test_svd():
    print(f'# TEST: in {sys._getframe().f_code.co_name}')
    np.set_printoptions(precision=3, suppress=True)
    c = np.array([[1,1,1], [1,0,0], [1,0,0], [0,1,1], [0,1,0], [0,0,1]])
    svd = SVD(c)
    svd.execute()
    actual_Sigma = [2.3941701709713272, 1.505971179150226, 1.0000000000000002]
    for i, sig in enumerate(actual_Sigma):
        assert sig == svd.Sigma[i, i]
    actual_USVt = [ [ 1.0, 1.0, 1.0 ],
                    [ 1.0, 0.0, 0.0 ],
                    [ 1.0, 0.0, 0.0 ] ,
                    [ 0.0, 1.0, 1.0 ],
                    [ 0.0, 1.0, 0.0 ],
                    [ 0.0, 0.0, 1.0 ] ]
    expect_USVt = svd.calc_USVt()
    svd.check_matrix(actual_USVt, expect_USVt)
    assert True == svd.allclose()
    assert False == svd.array_equal()
    # print(f'# Element-Close := \n{svd.isclose()}')
    actual_USpVt = [ [ 1.0, 1.0, 1.0 ],
                     [ 1.0, 0.0, 0.0 ],
                     [ 1.0, 0.0, 0.0 ] ,
                     [ 0.0, 1.0, 1.0 ],
                     [ 0.0, 0.5, 0.5 ],
                     [ 0.0, 0.5, 0.5 ] ]
    k_use = 2
    expect_USpVt = svd.reduce_USVt(k_use)
    svd.check_matrix(actual_USpVt, expect_USpVt)


def matrix_test():
    m1 = np.matrix([[1, 2], [3, 4]])
    m2 = np.matrix([[1, 1], [1, 1]])
    print(m1 * m2)
    print(m1 ** -1)
    print(m1.I)
    m3 = np.matrix([[1, 3], [-2, -4]])
    Pdiag = (np.linalg.eig(m3))[0]
    P = (np.linalg.eig(m3))[1]
    print(Pdiag)
    print(P)
    PiAP = P.I * m3 * P
    PiAP = PiAP.round(3)
    print(PiAP)
    print((PiAP)**2)
    print((PiAP)**3)


def array_test():
    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([[1, 1], [1, 1]])
    print(a1 * a2)
    print(a1.dot(a2))

    a1 = np.ones([2, 2], dtype=int)
    a2 = np.arange(1, 5).reshape(2, 2)
    print(a1)
    print(a2)
    print(np.dot(a1, a2))
    print(a1.dot(a2))
    print(np.dot(a2, a1))

    a1 = np.array([[1, 2], [3, 4]])
    print(a1)
    print(a1.T)
    print(np.diag(a1))
    print(np.diag([1, 2, 3, 4, 5]))
    print(np.trace(a1))
    print(np.linalg.det(a1))
    print(np.linalg.eig(a1))
    print(np.linalg.inv(a1))
    print(np.linalg.solve(a1, [5, 13]))


if __name__ == '__main__':
    print(MathBasic.fibo_seq.__doc__)
    print(MathBasic.hypertan.__doc__)
    print(MathBasic.softmax.__doc__)
    print(f'{MathBasic.fibo_seq(100)}')
    for x in range(-20, 21, 2):
        print(x, MathBasic.hypertan(x))
    xs = np.arange(0, 5)
    print(xs.size, MathBasic.softmax(xs))
    print('# [2020/01/07] ----------------------------------')
    arr = np.asarray([0, 1, 1, 0, 1, 0, 0, 0], dtype=np.float32)
    print(arr)
    print(arr.shape)
    c = Counter(arr)
    print(c)
    print('# [2020/01/08] ----------------------------------')
    # main_random()
    print('# [2020/01/09] ----------------------------------')
    test_statistics()
    print('# [2020/01/21] ----------------------------------')
    # https://ohke.hateblo.jp/entry/2017/12/14/230500
    test_svd()
    print('# [2020/02/09] ----------------------------------')
    array_test()
    # matrix_test()
    
    # nbase = 10000 # 10min for diag
    # nbase = 100
    # a1 = np.diag([1.0 for x in range(nbase)])
    # r1 = np.random.rand(nbase, nbase)
    # print(a1)
    # print(r1)
    # for ir in range(a1.shape[0]):
    #     for jc in range(a1.shape[1]):
    #         if (ir < jc):
    #             a1[ir, jc] = r1[ir, jc]
    #             a1[jc, ir] = a1[ir, jc]
    # eigvec = np.linalg.eig(a1)
    # print(a1)
    # print(eigvec)
