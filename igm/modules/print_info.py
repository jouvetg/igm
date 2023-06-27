#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os
import datetime

def params_print_info(self):
    pass

def init_print_info(params,self):
 
    # Print parameters in screen and a dedicated file
    with open(
        os.path.join(params.working_dir, "igm-run-parameters.txt"), "w"
    ) as f:
        print("PARAMETERS ARE ...... ")
        for ck in params.__dict__:
            print("%30s : %s" % (ck, params.__dict__[ck]))
            print("%30s : %s" % (ck, params.__dict__[ck]), file=f)
            
    os.system('echo rm '+ os.path.join(params.working_dir, "igm-run-parameters.txt") + ' >> clean.sh')
    os.system('echo rm '+ os.path.join(params.working_dir, "computational-statistics.txt") + ' >> clean.sh')
    
    print("IGM %s :         Iterations   |         Time (y)     |     Time Step (y)   |   Ice Volume (km^3) ")

def update_print_info(params,self):
    """
    This serves to print key info on the fly during computation
    """
    if self.saveresult:
        print(
            "IGM %s :      %6.0f    |      %8.0f        |     %7.2f        |     %10.2f "
            % (
                datetime.datetime.now().strftime("%H:%M:%S"),
                self.it,
                self.t,
                self.dt_target,
                np.sum(self.thk) * (self.dx ** 2) / 10 ** 9,
            )
        )

def update_print_all_comp_info(params,self):
    """
    This serves to print computational info report
    """

    self.tcomp["all"] = []
    self.tcomp["all"].append(np.sum([np.sum(self.tcomp[f]) for f in self.tcomp.keys()])) 

    print("Computational statistics report:")
    with open(
        os.path.join(params.working_dir, "computational-statistics.txt"), "w"
    ) as f:
        for key in self.tcomp.keys():
            CELA = (
                key,
                np.mean(self.tcomp[key]),
                np.sum(self.tcomp[key]),
                len(self.tcomp[key]),
            )
            print(
                "     %15s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it : %8.0f"
                % CELA,
                file=f,
            )
            print(
                "     %15s  |  mean time per it : %8.4f  |  total : %8.4f  |  number it  : %8.0f"
                % CELA
            )
  
