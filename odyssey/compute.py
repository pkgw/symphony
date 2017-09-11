#! /usr/bin/env python
#
# Compute some synchrotron coefficients with Symphony, sampling randomly in
# a predefined parameter space.

from __future__ import absolute_import, division, print_function, unicode_literals

import io, sys, time
import numpy as np
from pwkit import cgs
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


powerlaw_parameters = [
    Param('s', 0.07, 1e8, True),
    Param('theta', 0.001, 0.5 * np.pi, False),
    Param('p', 1.5, 4, False),
]

n_calcs = 1024

nu_ref = 1e9
ne_ref = 1.0

def main():
    distrib = sys.argv[1]
    outpath = sys.argv[2]

    if distrib == 'powerlaw':
        func = symphony.compute_all_nontrivial
        parameters = powerlaw_parameters
    else:
        raise ValueError('bad distrib: %r' % (distrib,))

    # on the distribution function being used.

    n_params = len(parameters)
    pvals = np.random.uniform(size=(n_calcs, n_params))
    for i in range(n_params):
        pvals[:,i] = parameters[i].unit_to_phys(pvals[:,i])

    # Pre-compute this

    assert parameters[0].name == 's', 'first param must be s for kwargs code to work'
    B = 2 * np.pi * nu_ref * cgs.me * cgs.c / (cgs.e * pvals[:,0])

    # We expect to time out and get killed before we finish all of our
    # calculations, which is why we line-buffer our output.

    with io.open(outpath, 'at', 1) as outfile:
        print('#', ' '.join(p.summary() for p in parameters), file=outfile)
        kwargs = {'eat_errors': True}

        for i in range(n_calcs):
            kwargs.update((parameters[j].name, pvals[i,j]) for j in range(1, n_params))
            info = func(nu_ref, B[i], ne_ref, **kwargs)
            if not np.all(np.isfinite(info)):
                print('WARNING: got nonfinite answers for:', ' '.join('%.18e' % p for p in pvals[i]), file=sys.stderr)
                sys.stderr.flush()
            vec = list(pvals[i]) + list(info)
            print('\t'.join('%g' % v for v in vec), file=outfile)


if __name__ == '__main__':
    main()
