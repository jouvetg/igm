#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

from igm.modules.process.iceflow import iceflow
from igm.modules.process.time import time
from igm.modules.process.thk import thk


def params(parser):
    iceflow.params(parser)
    time.params(parser)
    thk.params(parser)


def initialize(params, state):
    iceflow.initialize(params, state)
    time.initialize(params, state)
    thk.initialize(params, state)


def update(params, state):
    iceflow.update(params, state)
    time.update(params, state)
    thk.update(params, state)


def finalize(params, state):
    iceflow.finalize(params, state)
    time.finalize(params, state)
    thk.finalize(params, state)
