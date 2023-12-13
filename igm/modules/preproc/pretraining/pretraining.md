
### <h1 align="center" id="title">IGM module `pretaining` </h1>

# Description:

This module performs a pretraining of the ice flow iflo_emulator on a glacier catalogue to improve the performance of the emaulator when used in glacier forward run. The pretraining can be relatively computationally demanding task (a couple of hours). This module should be called alone independently of any other igm module. Here is an example of paramter file:

```json
{
  "modules_preproc": ["pretraining"],
  "modules_process": [],
  "modules_postproc": [],
  "data_dir": "surflib3d_shape_100",
  "iflo_solve_nbitmax": 2000,
  "iflo_solve_stop_if_no_decrease": false,
  "iflo_retrain_emulator_lr": 0.0001,
  "iflo_dim_arrhenius": 3,
  "soft_begining": 500
}
```

 To run it, one first needs to have available a glacier catalogue.
 I provide here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8332898.svg)](https://doi.org/10.5281/zenodo.8332898) a dataset of a glacier catalogue (mountain glaciers) I have mostly used for pretraining IGM emaulators.

Once downloaded (or self generated), the folder 
"surflib3d_shape_100" can be re-organized into a subfolder "train" and a subfolder "test"  as follows:

```
├── test
│   └── NZ000_A78_C0
└── train
    ├── ALP02_A78_C0
    ├── ALP03_A78_C0
    ├── ALP04_A78_C0
    ├── ALP05_A78_C0
    ├── ALP06_A78_C0
    ├── ALP11_A78_C0
    ├── ALP17_A78_C0
```

The path (or name of the data folder) must be pass in parameter `data_dir`.
