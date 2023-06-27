#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 14:06:31 2023

@author: gjouvet

"""
 
import argparse 

def params_core():
    
    # this create core parameters
    parser = argparse.ArgumentParser(description="IGM")
    
    parser.add_argument(
        "--tstart", 
        type=float, 
        default=2000.0, 
        help="Start modelling time (default 2000)"
    )
    parser.add_argument(
        "--tend", 
        type=float, 
        default=2100.0, 
        help="End modelling time (default: 2100)"
    )
    parser.add_argument(
        "--working_dir", 
        type=str, default="", 
        help="Working directory (default empty string)"
    )
    parser.add_argument(
        "--logging_level", 
        type=str, default="CRITICAL", 
        help="Looging level"
    )
    parser.add_argument(
        "--logging_file", 
        type=str, default="igm.log", 
        help="Looging level"
    )
    
    return parser

    # # parsing ...
    # config = self.parser.parse_args()  