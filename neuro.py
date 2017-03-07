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
from collections import OrderedDict
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

config_path = os.path.join(os.path.dirname(__file__), 'nn_config.toml')


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


    def to_dict(self):
        d = OrderedDict()
        d['name'] = self.name
        d['negate'] = self.negate
        d['is_log'] = self.is_log
        d['mean'] = self.mean
        d['std'] = self.std
        d['phys_min'] = self.p_min
        d['phys_max'] = self.p_max
        return d


    @classmethod
    def from_dict(cls, info):
        # Blah total hack to override constructor.
        inst = cls('x', np.array([1., 2, 3]), False)
        inst.name = str(info['name'])
        inst.negate = bool(info['negate'])
        inst.is_log = bool(info['is_log'])
        inst.mean = float(info['mean'])
        inst.std = float(info['std'])
        inst.p_min = float(info['phys_min'])
        inst.p_max = float(info['phys_max'])

        # Figure out n_{min,max}. Do so manually just in case there ends up
        # being bounds checking that will blow up if n_{min,max} are unset.
        x = np.array([inst.p_min, inst.p_max])
        if inst.negate:
            x = -x
        if inst.is_log:
            x = np.log10(x)
        x = (x - inst.mean) / inst.std

        inst.n_min = x.min()
        inst.n_max = x.max()

        return inst


class DomainRange(object):
    n_params = len(hardcoded_params)
    n_results = hardcoded_n_results
    pmaps = None
    rmaps = None

    @classmethod
    def from_samples(cls, phys_samples):
        assert phys_samples.ndim == 2
        assert phys_samples.shape[1] == (cls.n_params + cls.n_results)

        inst = cls()
        inst.pmaps = []
        inst.rmaps = []

        for i in xrange(inst.n_params):
            inst.pmaps.append(Mapping(hardcoded_params[i][0], phys_samples[:,i], hardcoded_params[i][1]))

        for i in xrange(inst.n_results):
            inst.rmaps.append(Mapping('result%d' % i, phys_samples[:,i+inst.n_params], True))

        return inst


    @classmethod
    def from_serialized(cls, info):
        inst = cls()
        inst.pmaps = []
        inst.rmaps = []

        for subinfo in info['params']:
            inst.pmaps.append(Mapping.from_dict(subinfo))

        for subinfo in info['results']:
            inst.rmaps.append(Mapping.from_dict(subinfo))

        assert len(inst.pmaps) == cls.n_params
        assert len(inst.rmaps) == cls.n_results
        return inst


    @classmethod
    def from_config(cls):
        import pytoml

        with open(config_path) as f:
            cfg = pytoml.load(f)

        return cls.from_serialized(cfg)


    def __repr__(self):
        return '\n'.join(
            ['<%s n_p=%d n_r=%d' % (self.__class__.__name__, self.n_params, self.n_results)] +
            ['  P%d=%r,' % (i, m) for i, m in enumerate(self.pmaps)] +
            ['  R%d=%r,' % (i, m) for i, m in enumerate(self.rmaps)] +
            ['>'])


    def into_info(self, info):
        info['params'] = [m.to_dict() for m in self.pmaps]
        info['results'] = [m.to_dict() for m in self.rmaps]


class SampleData(object):
    domain_range = None
    phys = None
    norm = None

    def __init__(self, dirname):
        chunks = []

        for item in os.listdir(dirname):
            if not item.endswith('.txt'):
                continue

            c = np.loadtxt(os.path.join(dirname, item))
            if not c.size or c.ndim != 2:
                continue

            assert c.shape[1] == (DomainRange.n_params + DomainRange.n_results)
            chunks.append(c)

        self.phys = np.vstack(chunks)

        # Sadly necessary postprocessing. 1. Sometimes Symphony gives {j,alpha}_V > 0;
        # this should never happen with the range of thetas we explore.

        w = self.phys[:,9] >= 0
        self.phys[w,9] = np.nan
        w = self.phys[:,10] >= 0
        self.phys[w,10] = np.nan

        # End of postprocessing.

        self.domain_range = DomainRange.from_samples(self.phys)
        self.norm = np.empty_like(self.phys)

        for i in xrange(self.domain_range.n_params):
            self.norm[:,i] = self.domain_range.pmaps[i].phys_to_norm(self.phys[:,i])

        for i in xrange(self.domain_range.n_results):
            j = i + self.domain_range.n_params
            self.norm[:,j] = self.domain_range.rmaps[i].phys_to_norm(self.phys[:,j])

    @property
    def phys_params(self):
        return self.phys[:,:self.domain_range.n_params]

    @property
    def phys_results(self):
        return self.phys[:,self.domain_range.n_params:]

    @property
    def norm_params(self):
        return self.norm[:,:self.domain_range.n_params]

    @property
    def norm_results(self):
        return self.norm[:,self.domain_range.n_params:]


class NSModel(models.Sequential):
    """Neuro-Symphony Model -- just keras.models.Sequential extended with some
    helpers specific to our data structures.

    If initialized with `data`, a `SampleData` instance, you can train the
    neural net. Otherwise, you can just provide a `DomainRange` instance,
    which will fix how the input and output variables ("parameters" and
    "results") are normalized inside the neural net.

    """
    def __init__(self, result_index, domain_range, data=None):
        super(NSModel, self).__init__()
        self.result_index = int(result_index)
        self.domain_range = data.domain_range
        self.data = data
        assert self.result_index < self.domain_range.n_results


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
            pred = self.domain_range.rmaps[self.result_index].norm_to_phys(npred)
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
        self.data.phys[bad,self.domain_range.n_params+self.result_index] = np.nan
        self.data.norm[bad,self.domain_range.n_params+self.result_index] = np.nan
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
