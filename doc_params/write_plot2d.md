
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--editor_plot2d EDITOR_PLOT2D] [--plot_live PLOT_LIVE]
               [--plot_particles PLOT_PARTICLES] [--varplot VARPLOT] [--varplot_max VARPLOT_MAX]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--editor_plot2d`|`vs`|optimized for VS code (vs) or spyder (sp) for live plot|
||`--plot_live`||Display plots live the results during computation (Default: False)|
||`--plot_particles`||Display particles (Default: True)|
||`--varplot`|`velbar_mag`|variable to plot|
||`--varplot_max`|`250`|maximum value of the varplot variable used to adjust the scaling of the colorbar|
