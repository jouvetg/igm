
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--geology_file GEOLOGY_FILE] [--resample RESAMPLE] [--crop_data CROP_DATA]
               [--crop_xmin CROP_XMIN] [--crop_xmax CROP_XMAX] [--crop_ymin CROP_YMIN]
               [--crop_ymax CROP_YMAX]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--geology_file`|`geology.nc`|Input data file (default: geology.nc)|
||`--resample`|`1`|Resample the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points|
||`--crop_data`|`False`|Crop the data with xmin, xmax, ymin, ymax (default: False)|
||`--crop_xmin`|`None`|X left coordinate for cropping|
||`--crop_xmax`|`None`|X right coordinate for cropping|
||`--crop_ymin`|`None`|Y bottom coordinate fro cropping|
||`--crop_ymax`|`None`|Y top coordinate for cropping|
