#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Lock the rescaling scheme used to turn physical parameters into normalized
ones that behave well with the neural net.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from pwkit.io import Path
import pytoml
from symphony import neuro
import sys


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('datadir', type=str, metavar='DATADIR',
                    help='The path to the input sample data directory.')
    ap.add_argument('nndir', type=str, metavar='NNDIR',
                    help='The path to the output neural-net directory.')
    return ap




def main():
    args = make_parser().parse_args()

    # Load samples
    _, samples = neuro.basic_load(args.datadir)

    # Load skeleton config
    cfg_path = Path(args.nndir) / 'nn_config.toml'
    with cfg_path.open('rt') as f:
        info = pytoml.load(f)

    # Turn into processed DomainRange object
    dr = neuro.DomainRange.from_info_and_samples(info, samples)

    # Update config and rewrite
    dr.into_info(info)

    with cfg_path.open('wt') as f:
        pytoml.dump(f, info)


if __name__ == '__main__':
    main()
