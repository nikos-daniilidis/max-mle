from __future__ import print_function

from scipy.optimize import minimize
import numpy as np
from max_generator import MaxSelectGenerator
from likelihood_estimator import LikelihoodEstimator, VarianceLikelihoodEstimator


def get_data(num_obs=10000, num_gen=2, locs=[0., 0.], scales=[0.1, 0.12]):
    assert num_gen == len(locs)
    g = MaxSelectGenerator(locs=locs, scales=scales)
    es, ixs = g.select_events(num_obs)
    return es, ixs


def fit_variance_data(xs, ixs, num_gen=2, x0=[0.2, 0.2]):
    le = VarianceLikelihoodEstimator(xs=xs, ixs=ixs, num_generators=num_gen)
    result = minimize(fun=lambda x: le.log_likelihood(x)
                      , x0=x0)
                      #, bounds=[(-5., None), (-5., None)]
                      #, method="TNC")
                      #, jac=lambda x: -le.jacobian(x))
    return result


def fit_data(xs, ixs, num_gen=2, x0=[0., 0., 0.2, 0.2]):
    le = LikelihoodEstimator(xs=xs, ixs=ixs, num_generators=num_gen)
    result = minimize(fun=lambda x: le.log_likelihood(x)
                      , x0=x0
                      #, bounds=[(-5., None), (-5., None)]
                      #, method="Nelder-Mead"
                      , jac=lambda x: le.jacobian(x))
    return result


if __name__=="__main__":

    if False:
        # test variance likelihood estimator
        xs, ixs = get_data(num_obs=10000, num_gen=2, locs=[0.0, 0.], scales=[0.1, 0.12])
        s1 = np.mean(xs[ixs == 0])
        s2 = np.mean(xs[ixs == 1])
        print (s1, s2)
        result = fit_variance_data(xs, ixs, num_gen=2, x0=[np.log(2*s1), np.log(2*s2)])
        print (result)
        vle = VarianceLikelihoodEstimator(xs, ixs, 2)
        print (np.exp(result.x[0]), np.exp(result.x[1]))
        print (vle.log_likelihood([np.log(0.1), np.log(0.12)]))
        print (vle.log_likelihood([result.x[0], result.x[1]]))

    if True:
        # test likelihood estimator (mean and variance)
        xs, ixs = get_data(num_obs=100000, num_gen=2, locs=[0.05, 0.], scales=[0.1, 0.12])
        s1 = np.mean(xs[ixs == 0])
        s2 = np.mean(xs[ixs == 1])
        print(s1, s2)
        result = fit_data(xs, ixs, num_gen=2, x0=[0., 0., np.log(2 * s1), np.log(2 * s2)])
        print(result)
        le = LikelihoodEstimator(xs, ixs, 2)
        print(result.x[0], result.x[1], np.exp(result.x[2]), np.exp(result.x[3]))
        print(le.log_likelihood([0.05, 0., np.log(0.1), np.log(0.12)]))
        print(le.log_likelihood(result.x))
