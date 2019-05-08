from __future__ import print_function
from warnings import warn
import numpy as np


class MaxSelectGenerator(object):
    def __init__(self, locs, scales, base_event="normal"):
        """
        Initialize a sequence of event generators where each event follows some base distribution
        with loc and scale parameters.
        :param locs: list of float. The mean parameters of the event generators.
        :param scales: list of float. The standard deviation parameters of the event generators.
        :param base_event: str.
        """
        assert isinstance(locs, list)
        assert isinstance(scales, list)
        assert base_event in ["normal"]
        assert len(locs) == len(scales), "Length of locs must equal length of scales."
        self._num_generators = len(locs)
        self._locs = locs
        self._scales = scales
        if base_event == "normal":
            def _gen(n):
                return np.random.normal(loc=0, scale=1., size=n)
            self._gen = _gen
        else:
            warn("%s base_event is not supported. Reverting to uniform." % base_event)
            def _gen(n):
                return np.random.uniform(low=0., high=1., size=n)
            self._gen = _gen

    def get_all_events(self, n):
        """
        Generate n events for all the streams. For efficiency, generate n*_num_generators events
        following an underlying distribution, and offset/scale accordingly for each stream.
        :param n: int. The number of events.
        :return: numpy array. The events. Rows are event realizations, columns are generator streams.
        """
        x = np.reshape(self._gen(n*self._num_generators), newshape=(n, self._num_generators))
        v = np.array(self._scales)
        m = np.ones(shape=(n, self._num_generators)) * np.array(self._locs)
        return x * v + m

    def select_events(self, n):
        """
        Generate n events using get_all_events. For each event, select the maximum.
        :param n: int. The number of events.
        :return: tuple (numpy array, numpy array). The scores and stream indices for the maxima.
        """
        es = self.get_all_events(n)
        ixs = np.argmax(es, axis=1)
        return es[np.arange(0, len(ixs), 1), ixs], ixs

    def locs(self):
        return self._locs

    def scales(self):
        return self._scales


if __name__ == "__main__":
    # check that locs work as expected
    #g = MaxSelectGenerator(locs=[0., 1., 2.], scales=[0.01, 0.01, 0.01])
    #es = g.get_all_events(10)
    #print(es)

    # check that scales work as expected
    g = MaxSelectGenerator(locs=[1., 1., 1.], scales=[0.001, 0.01, 0.1])
    # es = g.get_all_events(10)
    # print(es)

    es, ixs = g.select_events(10)
    print(ixs)
    print(es)
    print(es[ixs==0])