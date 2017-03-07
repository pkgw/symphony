#! /usr/bin/env python
# -*- mode: python; coding: utf-8 -*-
# Copyright 2017 Peter Williams and collaborators.
# Licensed under the MIT License.

"""Lock the rescaling scheme used to turn physica parameters into normalized
ones that behave well with the neural net.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from pwkit.io import Path
import pytoml
from symphony import neuro


def main():
    datadir = sys.argv[1]
    sd = neuro.SampleData(datadir)

    with Path(neuro.config_path).try_open(null_if_noexist=True) as f:
        cfg = pytoml.load(f)

    sd.domain_range.into_info(cfg)

    with Path(neuro.config_path).open('w') as f:
        pytoml.dump(f, cfg)


if __name__ == '__main__':
    main()
