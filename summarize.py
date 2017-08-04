#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Summarize the results from a Symphony sampling run

"""
from __future__ import absolute_import, division, print_function

import argparse, sys
import numpy as np
from pwkit.io import Path


def basic_load(datadir):
    datadir = Path(datadir)
    chunks = []
    param_names = None

    for item in datadir.glob('*.txt'):
        if param_names is None:
            with item.open('rt') as f:
                first_line = f.readline()
                assert first_line[0] == '#'
                param_names = first_line.strip().split()[1:]

        c = np.loadtxt(str(item))
        if not c.size or c.ndim != 2:
            continue

        assert c.shape[1] > len(param_names)
        chunks.append(c)

    data = np.vstack(chunks)
    return param_names, data


def analyze(datadir):
    param_names, data = basic_load(datadir)
    params = data[:,:len(param_names)]
    results = data[:,len(param_names):]

    # Report stuff.

    print('Number of parameter columns:', params.shape[1])
    print('Number of result columns:', results.shape[1])
    print('Number of rows:', data.shape[0])

    print('Total number of NaNs:', np.isnan(data).sum())
    print('Number of rows with NaNs:', (np.isnan(data).sum(axis=1) > 0).sum())

    for i in range(results.shape[1]):
        r = results[:,i]
        print()
        print('Result %d:' % i)
        print('  Number of NaNs:', np.isnan(r).sum())
        print('  Non-NaN max:', np.nanmax(r))
        print('  Non-NaN min:', np.nanmin(r))
        print('  Nonnegative:', (r >= 0).sum())
        print('  Nonpositive:', (r <= 0).sum())


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('datadir', type=str, metavar='DATADIR',
                    help='The path to the sample data directory.')
    return ap


def main():
    args = make_parser().parse_args()
    analyze(args.datadir)


if __name__ == '__main__':
    main()
