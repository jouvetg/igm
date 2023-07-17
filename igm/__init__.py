#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

from igm.state import *
from igm.params_core import *

from igm.modules.utils import *

from igm.modules.prepare_data import *
from igm.modules.make_synthetic import *
from igm.modules.load_ncdf_data import *
from igm.modules.load_tif_data import *

from igm.modules.optimize_v1 import *
from igm.modules.optimize import *

from igm.modules.iceflow_v1 import *
from igm.modules.iceflow import *
from igm.modules.flow_dt_thk import *
from igm.modules.vertical_iceflow import *
from igm.modules.smb_simple import *
from igm.modules.thk import *
from igm.modules.time_step import *
from igm.modules.particles_v1 import *
from igm.modules.particles import *
from igm.modules.topg_glacial_erosion import *
from igm.modules.rockflow import *
from igm.modules.print_info import *

from igm.modules.write_tif_ex import *
from igm.modules.write_ncdf_ex import *
from igm.modules.write_ncdf_ts import *
from igm.modules.write_plot2d  import *
from igm.modules.write_particles import *

from igm.modules.anim3d_from_ncdf_ex import *
from igm.modules.anim_mp4_from_ncdf_ex import *
from igm.modules.print_all_comp_info import *
