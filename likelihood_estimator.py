from __future__ import print_function

from collections import defaultdict
from scipy.stats import norm
from scipy.special import erf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class LikelihoodEstimator(object):
    def __init__(self, xs, ixs, num_generators):
        """
        Initialize a likelihood estimator with a set of observations. xs and ixs are the observed data.
        :param xs: numpy array. The observed event values.
        :param ixs: numpy array. The observed indices.
        :param num_generators: int. The number of streams (categories) from which events are selected.
        """
        assert isinstance(xs, (list, tuple, np.ndarray))
        assert isinstance(ixs, (list, tuple, np.ndarray))
        assert isinstance(num_generators, int)
        assert len(xs) == len(ixs), "Length of xs must equal length of ixs."
        self._n = num_generators
        self._xs = xs
        self._ixs = ixs

    def log_likelihood(self, params):
        """
        Return the log likelihood for given values of the mean and variance parameters. params is
        a concatenation of the mean and variance parameter vectors (length 2*num_generators)
        :param params: array_like, shape (n,). The vector of mean and variance parameters.
        :return: float. The log likelihood.
        """
        assert len(params) == 2 * self._n
        locs, scales = params[:self._n], np.exp(params[self._n:])
        st = set(self._ixs)
        ll = 0
        for ix in st:
            xix = self._xs[self._ixs==ix]
            ll += np.sum(np.log(norm.pdf(xix, loc=locs[ix], scale=scales[ix])))
            for jx in st:
                if jx != ix:
                    ll += np.sum(np.log(norm.cdf(xix, loc=locs[jx], scale=scales[jx])))

        return -ll

    def jacobian(self, params):
        """
        Return the Jacobian of the log likelihood with respect to the mean and log variance parameters.
        params is a concatenation of the mean and log variance parameter vectors (length 2*num_generators).
        :param params: array_like, shape (2*n,). The vector of mean and log variance parameters.
        :return: array_like, shape (n,). The Jacobian.
        """
        assert len(params) == 2 * self._n
        locs, scales = params[:self._n], np.exp(params[self._n:])
        st = set(self._ixs)
        dl = defaultdict(int)
        for ix in st:
            xix = self._xs[self._ixs == ix]
            xother = self._xs[self._ixs != ix]
            dl[ix] = self._dl_dm(xl=xix, xo=xother, loc=locs[ix], scale=scales[ix])
            dl[self._n + ix] = scales[ix] * self._dl_ds(xl=xix, xo=xother, loc=locs[ix], scale=scales[ix])
        return -np.array([dl[ix] for ix in range(2*self._n)])

    def _dl_dm(self, xl, xo, loc, scale):
        """
        Derivative of log likelihood with respect to mean
        \sum_{k: i_k=L} d(log(pdf(x_k; loc[L], scale[L])))/dloc[L]
        + \sum_{k: i_k!=L} d(log(cdf(x_k; loc[L], scale[L])))/dloc[L]
        :param xix: array_like, shape (nk, ). The xs coming from the stream with loc, value parameters.
        :param xo: array_like, shape (nother, ). The xs coming from all other streams.
        :param loc: array_like, shape (n,). The vector of mean parameters.
        :param scale: array_like, shape (n,). The vector of log variance parameters.
        :return: float
        """
        return (
            np.sum((xl - loc) / scale ** 2) -
            np.sum(np.exp(-(xo - loc)**2 / (2 * scale**2))
                    /(np.sqrt(2 * np.pi) * scale * (erf((xo-loc) / (np.sqrt(2) * scale)) / 2 + 1./2))
            )
        )

    def _dl_ds(self, xl, xo, loc, scale):
        """
        Derivative of log likelihood with respect to scale
        \sum_{k: i_k=L} d(log(pdf(x_k; loc[L], scale[L])))/dloc[L]
        + \sum_{k: i_k!=L} d(log(cdf(x_k; loc[L], scale[L])))/dloc[L]
        :param xix: array_like, shape (nk, ). The xs coming from the stream with loc, value parameters.
        :param xo: array_like, shape (nother, ). The xs coming from all other streams.
        :param loc: array_like, shape (n,). The vector of mean parameters.
        :param scale: array_like, shape (n,). The vector of log variance parameters.
        :return: float
        """
        return (
            np.sum(((xl-loc)**2) / scale**3 - 1 / scale) -
            np.sum(
                ((xo - loc) * np.exp(-(xo - loc)**2 / (2 * scale**2)))
                    / (np.sqrt(2 * np.pi) * scale**2 * (erf((xo - loc) / (np.sqrt(2) * scale)) / 2 + 1./2))
            )
        )


class VarianceLikelihoodEstimator(LikelihoodEstimator):
    def __init__(self, xs, ixs, num_generators, locs):
        """
        Initialize a likelihood estimator with a set of observations. xs and ixs are the observed data.
        :param xs: numpy array. The observed event values.
        :param ixs: numpy array. The observed indices.
        :param num_generators: int. The number of streams (categories) from which events are selected.
        :param locs: array_like, shape (n,). The vector of mean parameters.
        """
        assert len(locs) == num_generators
        LikelihoodEstimator.__init__(self, xs, ixs, num_generators)
        self._locs = locs

    def log_likelihood(self, params):
        """
        Return the log likelihood for given values of the mean and variance parameters. params is
        a concatenation of the mean and variance parameter vectors (length 2*num_generators)
        :param params: array_like, shape (n,). The vector of log variance parameters.
        :return: float. The log likelihood.
        """
        assert len(params) == self._n
        scales = np.exp(params[:self._n])
        st = set(self._ixs)
        ll = 0
        for ix in st:
            xix = self._xs[self._ixs==ix]
            ll += np.sum(np.log(norm.pdf(xix, loc=self._locs[ix], scale=scales[ix])))
            for jx in st:
                if jx != ix:
                    ll += np.sum(np.log(norm.cdf(xix, loc=self._locs[jx], scale=scales[jx])))

        return -ll

    def jacobian(self, params):
        """
        Return the Jacobian of the log likelihood with respect to the variance parameters.
        params is the log of the variance parameter vector (length num_generators).
        :param params: array_like, shape (n,). The vector of log variance parameters.
        :return: array_like, shape (n,). The Jacobian with respect to log variance
        """
        assert len(params) == self._n
        scales = np.exp(params[:self._n])
        st = set(self._ixs)
        dl = defaultdict(int)
        for ix in st:
            xix = self._xs[self._ixs == ix]
            xother = self._xs[self._ixs != ix]
            dl[ix] = scales[ix] * self._dl_ds(xl=xix, xo=xother, loc=self._locs[ix], scale=scales[ix])
        return -np.array([dl[ix] for ix in range(self._n)])


class MeanLikelihoodEstimator(LikelihoodEstimator):
    def __init__(self, xs, ixs, num_generators, log_scales):
        """
        Initialize a likelihood estimator with a set of observations. xs and ixs are the observed data.
        :param xs: numpy array. The observed event values.
        :param ixs: numpy array. The observed indices.
        :param num_generators: int. The number of streams (categories) from which events are selected.
        """
        assert len(log_scales) == num_generators
        LikelihoodEstimator.__init__(self, xs, ixs, num_generators)
        self._scales = np.exp(log_scales)

    def log_likelihood(self, params):
        """
        Return the log likelihood for given values of the mean and variance parameters. params is
        a concatenation of the mean and variance parameter vectors (length 2*num_generators)
        :param params: array_like, shape (n,). The vector of mean parameters.
        :return: float. The log likelihood.
        """
        assert len(params) == self._n
        locs = params[:self._n]
        st = set(self._ixs)
        ll = 0
        for ix in st:
            xix = self._xs[self._ixs==ix]
            ll += np.sum(np.log(norm.pdf(xix, loc=locs[ix], scale=self._scales[ix])))
            for jx in st:
                if jx != ix:
                    ll += np.sum(np.log(norm.cdf(xix, loc=locs[jx], scale=self._scales[jx])))

        return -ll

    def jacobian(self, params):
        """
        Return the Jacobian of the log likelihood with respect to the variance parameters.
        params is the log of the variance parameter vector (length num_generators).
        :param params: array_like, shape (n,). The vector of mean parameters.
        :return: array_like, shape (n,). The Jacobian with respect to mean
        """
        assert len(params) == self._n
        locs = params[:self._n]
        st = set(self._ixs)
        dl = defaultdict(int)
        for ix in st:
            xix = self._xs[self._ixs == ix]
            xother = self._xs[self._ixs != ix]
            dl[ix] = self._dl_dm(xl=xix, xo=xother, loc=locs[ix], scale=self._scales[ix])
        return -np.array([dl[ix] for ix in range(self._n)])


def test_likelihood_estimator_vars(es, ixs):
    sd1 = np.linspace(0.01, 0.4, num=20)
    sd2 = np.linspace(0.01, 0.4, num=20)
    lp = np.zeros([len(sd1), len(sd2)])
    for ix, s1 in enumerate(sd1):
        for jx, s2 in enumerate(sd2):
            le = LikelihoodEstimator(xs=es, ixs=ixs, num_generators=2)
            vle = VarianceLikelihoodEstimator(xs=es, ixs=ixs, num_generators=2, locs=[0., 0.])
            # lp[ix, jx] = le.log_likelihood(params=[0., 0., np.log(s1), np.log(s2)])
            lp[ix, jx] = vle.log_likelihood(params=[np.log(s1), np.log(s2)])

    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca(projection='3d')
    S1, S2 = np.meshgrid(sd1, sd2)
    surf = ax.plot_surface(S1, S2, lp)
    ax = fig.gca()
    ax.contour(sd1, sd2, lp)
    plt.show()
    i = np.argmax(lp, axis=0)
    j = np.argmax(lp[i, :], axis=1)
    print(sd1[i], sd2[j])


def test_likelihood_estimator_meanvar(es, ixs):
    ms1 = np.linspace(-0.05, 0.05, num=20)
    sd1 = np.linspace(0.09, 0.13, num=20)
    lp = np.zeros([len(ms1), len(sd1)])
    for ix, m1 in enumerate(ms1):
        for jx, s1 in enumerate(sd1):
            le = LikelihoodEstimator(xs=es, ixs=ixs, num_generators=2)
            lp[ix, jx] = le.log_likelihood(params=[m1, 0., s1, 0.12])

    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca(projection='3d')
    M1, S1 = np.meshgrid(ms1, sd1)
    surf = ax.plot_surface(M1, S1, lp)
    ax = fig.gca()
    ax.contour(ms1, sd1, lp)
    plt.show()
    i = np.argmax(lp, axis=0)
    j = np.argmax(lp[i, :], axis=1)
    print(ms1[i], sd1[j])


def test_likelihood_all(es, ixs):
    le = LikelihoodEstimator(xs=es, ixs=ixs, num_generators=2)
    vle = VarianceLikelihoodEstimator(xs=es, ixs=ixs, num_generators=2, locs=[0.05, 0.])
    mle = MeanLikelihoodEstimator(xs=es, ixs=ixs, num_generators=2, log_scales=[np.log(0.1), np.log(0.12)])
    print(le.log_likelihood([0.05, 0., np.log(0.1), np.log(0.12)]))
    print(vle.log_likelihood([np.log(0.1), np.log(0.12)]))
    print(mle.log_likelihood([0.05, 0.]))


def test_likelihood_estimator_jacobian(es, ixs):
    le = LikelihoodEstimator(xs=es, ixs=ixs, num_generators=2)
    print(le.jacobian(params=[0.05, 0., np.log(0.1), np.log(0.12)]))
    print(le.jacobian(params=[0.05, 0., np.log(0.1), np.log(0.12)]).shape)
    print((le.log_likelihood(params=[0.0501, 0., np.log(0.1), np.log(0.12)]) -
           le.log_likelihood(params=[0.05, 0., np.log(0.1), np.log(0.12)])) / 0.0001)
    print((le.log_likelihood(params=[0.050, 0.0001, np.log(0.1), np.log(0.12)]) -
           le.log_likelihood(params=[0.05, 0., np.log(0.1), np.log(0.12)])) / 0.0001)
    print((le.log_likelihood(params=[0.050, 0.0, np.log(0.1)+0.0001, np.log(0.12)]) -
           le.log_likelihood(params=[0.05, 0., np.log(0.1), np.log(0.12)])) / 0.0001)
    print((le.log_likelihood(params=[0.050, 0.0, np.log(0.1), np.log(0.12)+0.0001]) -
           le.log_likelihood(params=[0.05, 0., np.log(0.1), np.log(0.12)])) / 0.0001)


def test_variance_likelihood_estimator_jacobian(es, ixs):
    vle = VarianceLikelihoodEstimator(xs=es, ixs=ixs, num_generators=2, locs=[0.05, 0.])
    print(vle.jacobian(params=[np.log(0.1), np.log(0.12)]))
    print(vle.jacobian(params=[np.log(0.1), np.log(0.12)]).shape)
    print((vle.log_likelihood(params=[np.log(0.1)+0.0001, np.log(0.12)]) -
           vle.log_likelihood(params=[np.log(0.1), np.log(0.12)])) / 0.0001)
    print((vle.log_likelihood(params=[np.log(0.1), np.log(0.12)+0.0001]) -
           vle.log_likelihood(params=[np.log(0.1), np.log(0.12)])) / 0.0001)


def test_mean_likelihood_estimator_jacobian(es, ixs):
    mle = MeanLikelihoodEstimator(xs=es, ixs=ixs, num_generators=2, log_scales=[np.log(0.1), np.log(0.12)])
    print(mle.jacobian(params=[0., 0.]))
    print(mle.jacobian(params=[0., 0.]).shape)
    print((mle.log_likelihood(params=[0.000001, 0.]) -
           mle.log_likelihood(params=[0., 0.])) / 0.000001)
    print((mle.log_likelihood(params=[0., 0.000001]) -
           mle.log_likelihood(params=[0., 0.])) / 0.000001)


if __name__ == "__main__":
    from max_generator import MaxSelectGenerator

    g = MaxSelectGenerator(locs=[0., 0.], scales=[0.1, 0.12])
    es, ixs = g.select_events(100000)

    #test_likelihood_estimator_vars(es=es, ixs=ixs)

    #test_likelihood_estimator_meanvar(es=es, ixs=ixs)

    #test_likelihood_all(es=es, ixs=ixs)

    #test_likelihood_estimator_jacobian(es=es, ixs=ixs)

    test_variance_likelihood_estimator_jacobian(es=es, ixs=ixs)

    #test_mean_likelihood_estimator_jacobian(es=es, ixs=ixs)

