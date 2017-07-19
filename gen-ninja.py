#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

from __future__ import absolute_import, division, print_function

import os, sys
from pwkit import io, ninja_syntax


def inner(w, cc, mode):
    # Regenerating build.ninja

    w.comment('Automatically generated.')

    w.rule('regen',
           command = './gen-ninja.py %s %s' % (cc, mode),
           description = 'GEN $out',
           generator = True,
    )

    w.build('build.ninja', 'regen',
            implicit = 'gen-ninja.py',
    )

    # Assemble various args

    import numpy as np
    from distutils import sysconfig

    stack_path = '../stack'
    abs_stack_path = str(io.Path(stack_path).resolve())

    cpp_includes = [
        '-Isrc',
        '-I%s/include' % stack_path,
        '-I' + np.get_include(),
        '-I' + sysconfig.get_python_inc(),
    ]

    cpp_args = cpp_includes + ['-Dsymphony_EXPORTS']

    cc_args = [
        '-std=c99',
        '-g',
        '-fPIC',
        '-Wall',
    ]

    if mode == 'debug':
        cc_args += ['-O0']
    else:
        cc_args += ['-O3']

    ld_args = [
        '-fPIC',
        '-shared',
        '-Wl,-rpath,%s/lib' % abs_stack_path,
    ]

    libraries = [
        '-L%s/lib' % stack_path,
        '-lgsl',
        '-lgslcblas',
    ]

    w.rule('cc',
           command = '%s %s %s -MD -MF $out.d -o $out -c $in' %
               (cc, ' '.join(cpp_args), ' '.join(cc_args)),
           description = 'CC $out',
           deps = 'gcc',
           depfile = '$out.d',
    )

    # C compilation.

    c_sources = [
        'src/bessel_mod.c',
        'src/demo.c',
        'src/distribution_function_common_routines.c',
        'src/fits.c',
        'src/integrator/integrands.c',
        'src/integrator/integrate.c',
        'src/kappa/kappa.c',
        'src/kappa/kappa_fits.c',
        'src/maxwell_juettner/maxwell_juettner.c',
        'src/maxwell_juettner/maxwell_juettner_fits.c',
        'src/params.c',
        'src/pkgw_pitchy_power_law.c',
        'src/power_law/power_law.c',
        'src/power_law/power_law_fits.c',
        'src/symphony.c',
    ]


    w.rule('cython',
           command = 'cython %s -2 --output-file $out $in' %
               (' '.join(cpp_includes),),
           description = 'CYTHON $out',
    )

    w.build('symphonyPy.c', 'cython',
            inputs = ['src/symphonyPy.pyx'],
            implicit = ['src/symphonyHeaders.pxd'],
    )
    c_sources.append('symphonyPy.c')

    objects = []

    for c in c_sources:
        obj = c.replace('.c', '.o')
        objects.append(obj)
        w.build(obj, 'cc', inputs=[c])

    # Linking

    w.rule('link',
           command = '%s %s -o $out $in %s' % (cc, ' '.join(ld_args), ' '.join(libraries)),
           description = 'LINK $out',
    )

    w.build('symphonyPy.so', 'link', inputs = objects)


def outer(args):
    me = io.Path(sys.argv[0]).parent

    cc = args[0]
    mode = args[1]

    if mode not in ('debug', 'prod'):
        raise Exception('second arg must be "debug" or "prod"')

    with (me / 'build.ninja').open('wt') as f:
        w = ninja_syntax.Writer(f)
        inner(w, cc, mode)


if __name__ == '__main__':
    import sys
    outer(sys.argv[1:])
