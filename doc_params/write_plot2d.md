### <h1 align="center" id="title">IGM module write_plt2d </h1>

# Description:

This IGM module produces 2D plan-view plots of variable params.varplot at
a given frequency. The plots are saved as png files in the working directory.

# I/O:

Input: variable to be plotted
Output: png files


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--editor_plot2d`|`vs`|optimized for VS code (vs) or spyder (sp) for live plot|
||`--plot_live`||Display plots live the results during computation (Default: False)|
||`--plot_particles`||Display particles (Default: True)|
||`--varplot`|`velbar_mag`|variable to plot|
||`--varplot_max`|`250`|maximum value of the varplot variable used to adjust the scaling of the colorbar|
 
# Parameters: 
