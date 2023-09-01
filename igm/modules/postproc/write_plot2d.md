### <h1 align="center" id="title">IGM module `write_plot2d` </h1>

# Description:

This IGM module produces 2D plan-view plots of variable defined by parameter `varplot` (e.g. `varplot` can be set to `thk`, or `ubar`, ...). The saving frequency is given by parameter `time_save` defined in module `time_step`.  The scale range of the colobar is controlled by parameter `varplot_max`.

By default, the plots are saved as png files in the working directory. However, one may display the plot "in live" by setting `plot_live` to True. Note that if you use the spyder python editor, you need to turn `editor_plot2d` to 'sp'.
 
If the `particles` module is activated, one may plot particles on the top setting `plot_particles` to True, or remove them form the plot seeting it to False.
