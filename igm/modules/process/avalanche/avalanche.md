### <h1 align="center" id="title">IGM avalanche module  </h1>

# Description:

This IGM module permits to model redistribution of snow due to avalanches.
This routine move ice/snow downslope until the ice surface is everywhere
at angle of repose. This function was adapted from 
[Mark Kessler's GC2D](https://github.com/csdms-contrib/gc2d)
program and implemented in IGM by Jürgen Mey with support from Guillaume Jouvet.
 
Modified exit strategy, annotated and documented by Andreas Henz

The redistribution of snow is done in each specific time step `aval_update_freq` and based on a maximum slope angle `aval_angleOfRepose`. Basically, the ice is redistributed downwards in an iterative process to achieve a maximum surface slope (in the code a maximum height difference between neighboring grid cells). This iterative process is stopped when the average distributed ice is less than a certain thickness `aval_stop_redistribution_thk`. This initial value (recommended between 1-5 cm) was determined by looking at the amount of ice distributed at each time step. If very little ice is distributed, we stop the loop and save computation time.
