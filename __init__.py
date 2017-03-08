# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = str ('''
compute_coefficient
compute_all_nontrivial
''').split ()

import numpy as np
from . import symphonyPy

from .symphonyPy import STOKES_I, STOKES_Q, STOKES_U, STOKES_V
EMISSION, ABSORPTION = 0, 1


def compute_coefficient(
        rttype = EMISSION,
        stokes = STOKES_I,
        nu = 1e9, # you'll almost always want to override these but named params are nice.
        B = 100.,
        ne = 1e3,
        theta = 1.,
        p = 2.,
        gamma_min = 0.1,
        gamma_max = 1000.,
        gamma_cutoff = 1e7,
        approximate = False,
        eat_errors = False,
):
    """If you read the discussion in Pandya+2016, the Kappa distribution looks
    tempting, but I don't believe that anyone has computed Faraday
    coefficients for it.

    The approximate fitting formulae perform VERY badly if gamma_min <~ 1, it
    seems.

    """
    if ne == 0:
        # My code sometimes tries to get coefficients with n_e = p = 0, which
        # makes symphony barf; fortunately, if there's nothing there we know
        # exactly what every radiative transfer coefficient is:
        return 0.

    if rttype == EMISSION:
        if approximate:
            func = symphonyPy.j_nu_fit_py
        else:
            func = symphonyPy.j_nu_py
    elif rttype == ABSORPTION:
        if approximate:
            func = symphonyPy.alpha_nu_fit_py
        else:
            func = symphonyPy.alpha_nu_py
    else:
        raise ValueError ('unexpected value of "rttype": %r' % (rttype,))

    try:
        result = func (
            nu,
            B,
            ne,
            theta,
            symphonyPy.POWER_LAW,
            stokes,
            10., # Max/Jutt distribution: \Theta_e, dimensionless electron temperature
            p,
            gamma_min, # powerlaw distribution: gamma_min
            gamma_max, # powerlaw distribution: gamma_max
            gamma_cutoff, # powerlaw distribution: gamma_cutoff
            3.5, # kappa distribution: kappa
            10, # kappa distribution: kappa_width
            )
    except RuntimeError as e:
        if eat_errors:
            return np.nan
        raise

    return result


def compute_all_nontrivial(
        nu = 1e9, # you'll almost always want to override these but named params are nice.
        B = 100.,
        ne = 1e3,
        theta = 1.,
        p = 2.,
        gamma_min = 0.1,
        gamma_max = 1000.,
        gamma_cutoff = 1e7,
        approximate = False,
        eat_errors = False,
):
    result = np.empty(6)
    rest = (nu, B, ne, theta, p, gamma_min, gamma_max, gamma_cutoff, approximate, eat_errors)
    result[0] = compute_coefficient(EMISSION, STOKES_I, *rest)
    result[1] = compute_coefficient(ABSORPTION, STOKES_I, *rest)
    result[2] = compute_coefficient(EMISSION, STOKES_Q, *rest)
    result[3] = compute_coefficient(ABSORPTION, STOKES_Q, *rest)
    result[4] = compute_coefficient(EMISSION, STOKES_V, *rest)
    result[5] = compute_coefficient(ABSORPTION, STOKES_V, *rest)
    return result
