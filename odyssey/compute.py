#! /usr/bin/env python
#
# Compute some synchrotron coefficients with Symphony, sampling randomly in
# a predefined parameter space.

from __future__ import absolute_import, division, print_function, unicode_literals

import io, sys, time
import numpy as np
from six.moves import range
import symphony

class Param(object):
    def __init__(self, name, low, high, is_log):
        self.name = name
        self.low = float(low)
        self.high = float(high)
        self.is_log = bool(is_log)

        if self.is_log:
            self.low = np.log(self.low)
            self.high = np.log(self.high)

    def unit_to_phys(self, values):
        phys = values * (self.high - self.low) + self.low
        if self.is_log:
            phys = np.exp(phys)
        return phys

    def summary(self):
        return '%s(%s)' % (self.name, 'log' if self.is_log else 'lin')


parameters = [
    Param('nu', 1e8, 3e11, True),
    Param('B', 0.1, 1e4, True),
    Param('n_e', 1, 1e10, True),
    Param('theta', 0.001, 0.5 * np.pi, False),
    Param('p', 1.5, 4, False),
]

n_params = len(parameters)
n_calcs = 1024


def main():
    distrib = sys.argv[1]
    outpath = sys.argv[2]

    if distrib == 'powerlaw':
        func = symphony.compute_all_nontrivial
    elif distrib == 'pitchy':
        func = symphony.compute_all_nontrivial_pitchy
    else:
        raise ValueError('bad distrib: %r' % (distrib,))

    # TODO: might potentially have different parameter sets depending
    # on the distribution function being used.

    pvals = np.random.uniform(size=(n_calcs, n_params))
    for i in range(n_params):
        pvals[:,i] = parameters[i].unit_to_phys(pvals[:,i])

    # We expect to time out and get killed before we finish all of our
    # calculations, which is why we line-buffer our output.

    with io.open(outpath, 'at', 1) as outfile:
        print('#', ' '.join(p.summary() for p in parameters), file=outfile)

        for i in range(n_calcs):
            info = func(*pvals[i], eat_errors=True)
            vec = list(pvals[i]) + list(info)
            print('\t'.join('%g' % v for v in vec), file=outfile)


if __name__ == '__main__':
    main()
