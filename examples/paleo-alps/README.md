
# Overview

This set-up gives a simple set-up to run a paleo glacier model in the European Alps in paleo times with different catchements (lyon, ticino, rhine, linth glaciers) with IGM (the Instructed Glacier Model) e.g. around the last glacial maximum (LGM, 24 BP in the Alps).

Try to take different topography (copy from forlder data into topg.tif in the current directory) to experience diferent catchements.

# Inputs files

Input files are found in the folder data. There is:
 
a) Tiff files that contains the present-day topography after substracting present-day glaciers (dataset by Millan and al., 2022), and present-day lakes (Swisstopo data).

b) The EPICA climate temperature difference signal to drive the climate forcing (Ref: Jouzel, Jean; Masson-Delmotte, Valerie (2007): EPICA Dome C Ice Core 800KYr deuterium data and temperature estimates)

c) Some flowlines usefull to plot result in plot-result.py

# Usage
	
Make sure the IGM's dependent libraries ar installed, or activate your igm environment with conda

		conda activate igm
	 
You may change parameters in params.json, copy one of the topg-XXXX.tif in the folder after renaming it 'topg.tif', and then run igm with 

		python igm-run.py
		
Don't forget to clean behind you:

		sh clean.sh

# Vizualize results

After any run, you may plot some results with companion python scripts (plot-result.py), or vizualize results with `ncview output.nc`.

# References

@article{millan2022ice,
  title={Ice velocity and thickness of the world’s glaciers},
  author={Millan, Romain and Mouginot, J{\'e}r{\'e}mie and Rabatel, Antoine and Morlighem, Mathieu},
  journal={Nature Geoscience},
  volume={15},
  number={2},
  pages={124--129},
  year={2022},
  publisher={Nature Publishing Group}
}

@misc{jouzel2007edci,
 author={Jean {Jouzel} and Valerie {Masson-Delmotte}},
 title={{EPICA Dome C Ice Core 800KYr deuterium data and temperature estimates}},
 year={2007},
 doi={10.1594/PANGAEA.683655},
 url={https://doi.org/10.1594/PANGAEA.683655},
 type={data set},
 publisher={PANGAEA}
}
