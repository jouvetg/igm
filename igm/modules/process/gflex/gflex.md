### <h1 align="center" id="title">IGM isostasy_gfex module  </h1>

# Description:

This IGM module permits to model the isostasy or upward motion of the 
lithosphere when loaded with thick ice, it uses the 
[gflex](https://gmd.copernicus.org/articles/9/997/2016/)
python module writen by Andy Wickert.

This function was implemented in IGM by Jürgen Mey.
 
Parameters are the update frequency `gflex_update_freq` and the Elastic thickness [m] (Te) `gflex_default_Te`.

This module only runs on CPU, which may be an issue for treating very large arrays.
On the other hand, we do not expect a frequent update, therefore, this module should not be overall too consuming.



