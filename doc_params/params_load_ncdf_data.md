usage: PAR.py [-h] [--geology_file GEOLOGY_FILE] [--resample RESAMPLE]
              [--crop_data CROP_DATA] [--crop_xmin CROP_XMIN]
              [--crop_xmax CROP_XMAX] [--crop_ymin CROP_YMIN]
              [--crop_ymax CROP_YMAX]

optional arguments:
  -h, --help            show this help message and exit
  --geology_file GEOLOGY_FILE
                        Input data file (default: geology.nc)
  --resample RESAMPLE   Resample the data to a coarser resolution (default:
                        1), e.g. 2 would be twice coarser ignore data each 2
                        grid points
  --crop_data CROP_DATA
                        Crop the data with xmin, xmax, ymin, ymax (default:
                        False)
  --crop_xmin CROP_XMIN
                        X left coordinate for cropping
  --crop_xmax CROP_XMAX
                        X right coordinate for cropping
  --crop_ymin CROP_YMIN
                        Y bottom coordinate fro cropping
  --crop_ymax CROP_YMAX
                        Y top coordinate for cropping
