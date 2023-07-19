#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

from igm.state import *
from igm.params_core import *

from igm.modules.utils import *

from igm.modules.preproc.prepare_data import *
from igm.modules.preproc.make_synthetic import *
from igm.modules.preproc.load_ncdf_data import *
from igm.modules.preproc.load_tif_data import *

from igm.modules.physics.optimize_v1 import *
from igm.modules.physics.optimize import *
from igm.modules.physics.iceflow_v1 import *
from igm.modules.physics.iceflow import *
from igm.modules.physics.flow_dt_thk import *
from igm.modules.physics.vertical_iceflow import *
from igm.modules.physics.smb_simple import *
from igm.modules.physics.thk import *
from igm.modules.physics.time_step import *
from igm.modules.physics.particles_v1 import *
from igm.modules.physics.particles import *
from igm.modules.physics.topg_glacial_erosion import *
from igm.modules.physics.rockflow import *

from igm.modules.postproc.print_info import *
from igm.modules.postproc.print_all_comp_info import *
from igm.modules.postproc.write_tif_ex import *
from igm.modules.postproc.write_ncdf_ex import *
from igm.modules.postproc.write_ncdf_ts import *
from igm.modules.postproc.write_plot2d  import *
from igm.modules.postproc.write_particles import *
from igm.modules.postproc.anim3d_from_ncdf_ex import *
from igm.modules.postproc.anim_mp4_from_ncdf_ex import *

