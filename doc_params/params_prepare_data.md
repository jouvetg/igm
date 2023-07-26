usage: PAR.py [-h] [--RGI RGI] [--preprocess PREPROCESS] [--dx DX]
              [--border BORDER] [--thk_source THK_SOURCE]
              [--include_glathida INCLUDE_GLATHIDA]
              [--path_glathida PATH_GLATHIDA]
              [--output_geology OUTPUT_GEOLOGY]

optional arguments:
  -h, --help            show this help message and exit
  --RGI RGI             RGI ID
  --preprocess PREPROCESS
                        Use preprocessing
  --dx DX               Spatial resolution (need preprocess false to change
                        it)
  --border BORDER       Safe border margin (need preprocess false to change
                        it)
  --thk_source THK_SOURCE
                        millan_ice_thickness or consensus_ice_thickness in
                        geology.nc
  --include_glathida INCLUDE_GLATHIDA
                        Make observation file (for IGM inverse)
  --path_glathida PATH_GLATHIDA
                        Path where the Glathida Folder is store, so that you
                        don't need to redownload it at any use of the script
  --output_geology OUTPUT_GEOLOGY
                        Write prepared data into a geology file
