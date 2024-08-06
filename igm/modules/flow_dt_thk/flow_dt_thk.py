#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

from igm.modules.process.iceflow import iceflow
from igm.modules.process.time_igm import time_igm
from igm.modules.process.thk import thk


def params(parser):
    iceflow.params(parser)
    time_igm.params(parser)
    thk.params(parser)


def initialize(params, state):
    iceflow.initialize(params, state)
    time_igm.initialize(params, state)
    thk.initialize(params, state)


def update(params, state):
    iceflow.update(params, state)
    time_igm.update(params, state)
    thk.update(params, state)


def finalize(params, state):
    iceflow.finalize(params, state)
    time_igm.finalize(params, state)
    thk.finalize(params, state)
