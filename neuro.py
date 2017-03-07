# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Code to develop and use an artificial neural network approximation ("regression") of
Symphony's synchrotron coefficients.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''
''').split ()

import os.path
import numpy as np
from keras import models, layers, optimizers


hardcoded_params = [
    ('nu', True), # True = is sampled in log space
    ('B', True),
    ('ne', True),
    ('theta', False),
    ('p', False),
]

hardcoded_n_results = 6


class Mapping(object):
    def __init__(self, name, phys_samples, is_log):
        self.name = name

        phys_samples = phys_samples[np.isfinite(phys_samples)]
        assert phys_samples.size, 'no valid samples for %s' % name

        if (phys_samples > 0).all():
            self.negate = False
        elif (phys_samples < 0).all():
            self.negate = True
        else:
            assert False, 'vector %s is zero-crossing' % name

        self.p_min = phys_samples.min()
        self.p_max = phys_samples.max()

        if self.negate:
            phys_samples = -phys_samples

        self.is_log = bool(is_log)
        if self.is_log:
            phys_samples = np.log10(phys_samples)

        self.mean = phys_samples.mean()
        self.std = phys_samples.std()

        normed = (phys_samples - self.mean) / self.std
        self.n_min = normed.min()
        self.n_max = normed.max()


    def __repr__(self):
        return '<Mapping %s neg=%r log=%r mean=%r sd=%r>' % \
            (self.name, self.negate, self.is_log, self.mean, self.std)


    def phys_to_norm(self, phys):
        # TODO: (optional?) bounds checking!
        if self.negate:
            phys = -phys
        if self.is_log:
            phys = np.log10(phys)
        return (phys - self.mean) / self.std


    def norm_to_phys(self, norm):
        # TODO: (optional?) bounds checking!
        norm = norm * self.std + self.mean
        if self.is_log:
            norm = 10**norm
        if self.negate:
            norm = -norm
        return norm


class SampleData(object):
    n_params = len(hardcoded_params)
    n_results = hardcoded_n_results

    def __init__(self, dirname):
        chunks = []

        for item in os.listdir(dirname):
            if not item.endswith('.txt'):
                continue

            c = np.loadtxt(os.path.join(dirname, item))
            if not c.size or c.ndim != 2:
                continue

            assert c.shape[1] == (self.n_params + self.n_results), '%s %r' % (item, c)
            chunks.append(c)

        self.phys = np.vstack(chunks)

        self.pmaps = []
        self.rmaps = []
        self.norm = np.empty_like(self.phys)

        for i in xrange(self.n_params):
            self.pmaps.append(Mapping(hardcoded_params[i][0], self.phys[:,i], hardcoded_params[i][1]))
            self.norm[:,i] = self.pmaps[i].phys_to_norm(self.phys[:,i])

        for i in xrange(self.n_results):
            self.rmaps.append(Mapping('param%d' % i, self.phys[:,i+self.n_params], True))
            self.norm[:,i+self.n_params] = self.rmaps[i].phys_to_norm(self.phys[:,i+self.n_params])

    @property
    def phys_params(self):
        return self.phys[:,:self.n_params]

    @property
    def phys_results(self):
        return self.phys[:,self.n_params:]

    @property
    def norm_params(self):
        return self.norm[:,:self.n_params]

    @property
    def norm_results(self):
        return self.norm[:,self.n_params:]


class NSModel(models.Sequential):
    """Neuro-Symphony Model -- just keras.models.Sequential extended with some
    helpers specific to our data structures.

    """
    def __init__(self, data, result_index):
        super(NSModel, self).__init__()
        self.data = data
        self.result_index = int(result_index)
        assert self.result_index < self.data.n_results


    def ns_fit(self, **kwargs):
        nres = self.data.norm_results[:,self.result_index]
        ok = np.isfinite(nres)
        nres = nres[ok].reshape((-1, 1))
        npar = self.data.norm_params[ok]
        return self.fit(npar, nres, **kwargs)


    def ns_validate(self, filter=True, to_phys=True):
        if to_phys:
            par = self.data.phys_params
            res = self.data.phys_results[:,self.result_index]
        else:
            par = self.data.norm_params
            res = self.data.norm_results[:,self.result_index]

        npred = self.predict(self.data.norm_params)[:,0]

        if filter:
            ok = np.isfinite(res) & np.isfinite(npred)
            par = par[ok]
            res = res[ok]
            npred = npred[ok]

        if to_phys:
            pred = self.data.rmaps[self.result_index].norm_to_phys(npred)
        else:
            pred = npred

        return par, res, pred
