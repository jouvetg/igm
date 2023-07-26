#!/usr/bin/env python3

### <h1 align="center" id="title">IGM module smb_simple </h1>

# Description:

This IGM modules models a simple mass balance model  parametrized by ELA, ablation
and accumulation gradients, and max acuumulation from a given file mb_simple_file

# I/O

Input  : state.usurf
Output : state.smb


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--mb_update_freq`|`1`|Update the mass balance each X years (1)|
||`--mb_simple_file`|`mb_simple_param.txt`|Name of the imput file for the simple mass balance model|
 
# Parameters: 
