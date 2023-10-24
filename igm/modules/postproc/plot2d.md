### <h1 align="center" id="title">IGM module `plot2d` </h1>

# Description:

This IGM module produces 2D plan-view plots of variable defined by parameter `plt2d_var` (e.g. `plt2d_var` can be set to `thk`, or `ubar`, ...). The saving frequency is given by parameter `time_save` defined in module `time`.  The scale range of the colobar is controlled by parameter `plt2d_varmax`.

By default, the plots are saved as png files in the working directory. However, one may display the plot "in live" by setting `plt2d_live` to True. Note that if you use the spyder python editor, you need to turn `plt2d_editor` to 'sp'.
 
If the `particles` module is activated, one may plot particles on the top setting `plt2d_particles` to True, or remove them form the plot seeting it to False.
