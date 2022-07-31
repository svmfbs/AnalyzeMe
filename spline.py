import sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class InterpolateSpline():
    degree_splprep = 3

    def __init__(self, xstart, xstop, num_div, include_stop=True):
        self.point = num_div
        self.endpoint = include_stop
        self.Xs = np.linspace(xstart, xstop, num=num_div, endpoint=include_stop)

    def reset_Xs(self, xstart, xstop, num_div, include_stop=True):
        self.point = num_div
        self.endpoint = include_stop
        self.Xs = np.linspace(xstart, xstop, num=num_div, endpoint=include_stop)

    def set_kind(self, kind):
        self.kind_interp1d = kind

    @classmethod
    def linear1d(cls, X, xs, ys):
        f = interpolate.interp1d(xs, ys, kind="slinear")
        return f(X)

    @classmethod
    def quadratic1d(cls, X, xs, ys):
        f = interpolate.interp1d(xs, ys, kind="quadratic")
        return f(X)

    @classmethod
    def cubic1d(cls, X, xs, ys):
        f = interpolate.interp1d(xs, ys, kind="cubic")
        return f(X)

    @classmethod
    def Akima1d(cls, X, xs, ys):
        f = interpolate.Akima1DInterpolator(xs, ys)
        return f(X)

    def __splprep(self, xs, ys, degree):
        (tck, u) = interpolate.splprep([xs, ys], k=degree, s=0)
        u = np.linspace(0, 1, num=self.point, endpoint=self.endpoint) 
        spline = interpolate.splev(u, tck)
        return (spline[0], spline[1])

    def spline1d(self, xs, ys, iptype):
        Xs = self.Xs
        if iptype == 1:
            Ys = self.linear1d(Xs, xs, ys)
            return (Xs, Ys)
        elif iptype == 2:
            Ys = self.quadratic1d(Xs, xs, ys)
            return (Xs, Ys)
        elif iptype == 3:
            Ys = self.cubic1d(Xs, xs, ys)
            return (Xs, Ys)
        elif iptype == 4:
            Ys = self.Akima1d(Xs, xs, ys)
            return (Xs, Ys)
        elif iptype == 5:
            return self.__splprep(xs, ys, self.degree_splprep)
        else:
            return None


if __name__ == "__main__":
    #x座標に関して、重複、減少傾向を持つ座標群
    # xs = [-5, -5, -3, 2, 3, 0, -2]
    # ys = [6, 1, 6, 7, 1, -1, 0]

    #x方向に増加傾向である座標群
    xs = [-5, 0,  1,  3, 4, 6]
    ys = [-4, 2, -2, -4, 0, 4]
    points = 100
    degree = 3

    ip = InterpolateSpline(xs[0], xs[-1], points)
    ip.degree_splprep = 3
    (x1, y1) = ip.spline1d(xs, ys, 3)
    (x2, y2) = ip.spline1d(xs, ys, 4)
    (x3, y3) = ip.spline1d(xs, ys, 5)

    #グリッド線やラベルなどを付与しつつスプライン曲線をプロット
    plt.plot(xs, ys, 'ro', label="controlpoint")
    plt.plot(x1, y1, label="interp1d")
    plt.plot(x2, y2, label="Akima1DInterpolator")
    plt.plot(x3, y3, label="splprep")
    plt.title("spline")
    xlim = [-10, 10]
    ylim = [-10, 10]
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='lower right')
    plt.grid(which='major', color='black', linestyle='dotted')
    plt.xticks(list(filter(lambda x: x%1==0, np.arange(xlim[0], xlim[1]))))
    plt.yticks(list(filter(lambda x: x%1==0, np.arange(ylim[0], ylim[1]))))
    plt.show()
