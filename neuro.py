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

        # Sadly necessary postprocessing. 1. Sometimes Symphony gives alpha_V > 0;
        # this should never happen with the range of thetas we explore.

        w = self.phys[:,10] >= 0
        self.phys[w,10] = np.nan

        # End of postprocessing.

        self.pmaps = []
        self.rmaps = []
        self.norm = np.empty_like(self.phys)

        for i in xrange(self.n_params):
            self.pmaps.append(Mapping(hardcoded_params[i][0], self.phys[:,i], hardcoded_params[i][1]))
            self.norm[:,i] = self.pmaps[i].phys_to_norm(self.phys[:,i])

        for i in xrange(self.n_results):
            self.rmaps.append(Mapping('result%d' % i, self.phys[:,i+self.n_params], True))
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
        """Train this ANN model on the data in `self.data`. This function just
        takes care of extracting the right parameter and avoiding NaNs.

        """
        nres = self.data.norm_results[:,self.result_index]
        ok = np.isfinite(nres)
        nres = nres[ok].reshape((-1, 1))
        npar = self.data.norm_params[ok]
        return self.fit(npar, nres, **kwargs)


    def ns_validate(self, filter=True, to_phys=True):
        """Test this network by having it predict all of the values in our
        training sample. Returns `(params, actual, nn)`, where `params` is
        shape `(N, self.data.n_params)` and is the input parameters, `actual`
        is shape `(N,)` and is the actual values returned by Symphony, and
        `nn` is shape `(N,)` and is the values predicted by the neural net.

        If `filter` is true, the results will be filtered such that neither
        `actual` nor `nn` contain non-finite values.

        If `to_phys` is true, the values will be returned in the physical
        coordinate system. Otherwise they will be returned in the normalized
        coordinate system.

        """
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


    def ns_sigma_clip(self, n_norm_sigma):
        """Assuming that self is already a decent approximation of the input data,
        try to improve things by NaN-ing out any measurements that are extremely
        discrepant with our approximation -- under the assumption that these are
        cases where Symphony went haywise.

        Note that this destructively modifies `self.data`.

        `n_norm_sigma` is the threshold above which discrepant values are
        flagged. It is evaluated using the differences between the neural net
        prediction and the training data in the *normalized* coordinate
        system.

        Returns the number of flagged points.
        """
        nres = self.data.norm_results[:,self.result_index]
        npred = self.predict(self.data.norm_params)[:,0]
        err = npred - nres
        m = np.nanmean(err)
        s = np.nanstd(err)
        bad = (np.abs((err - m) / s) > n_norm_sigma)
        self.data.phys[bad,self.data.n_params+self.result_index] = np.nan
        self.data.norm[bad,self.data.n_params+self.result_index] = np.nan
        return bad.sum()


    def ns_plot(self, param_index, plot_err=False, to_phys=False, thin=100):
        """Make a diagnostic plot comparing the approximation to the "actual" results
        from Symphony."""
        import omega as om

        par, act, nn = self.ns_validate(filter=True, to_phys=to_phys)

        if plot_err:
            err = nn - act
            p = om.quickXY(par[::thin,param_index], err[::thin], 'Error', lines=0)
        else:
            p = om.quickXY(par[::thin,param_index], act[::thin], 'Full calc', lines=0)
            p.addXY(par[::thin,param_index], nn[::thin], 'Neural', lines=0)

        return p
