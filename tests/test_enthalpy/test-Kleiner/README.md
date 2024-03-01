[![License badge](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
### <h1 align="center" id="title"> Test of the Enthalpy model </h1>

This setup permits to lead experiments A and B from Kleiner and al. (2015), and to verify the results with the analalytical solution. For that purpose, you need to type

```bash
python expA.py
python expB.py
```

We are comparing the analytical solution to the data folder was taken from [PoLIM](https://github.com/WangYuzhe/PoLIM-Polythermal-Land-Ice-Model) described by Wang and al. (2020), which greatly helped to lead thes experiments.

# Reference

@article{kleiner2015enthalpy,
  title={Enthalpy benchmark experiments for numerical ice sheet models},
  author={Kleiner, Thomas and R{\"u}ckamp, Martin and Bondzio, Johannes H and Humbert, Angelika},
  journal={The Cryosphere},
  volume={9},
  number={1},
  pages={217--228},
  year={2015},
  publisher={Copernicus GmbH}
}

@article{wang2020two,
  title={A two-dimensional, higher-order, enthalpy-based thermomechanical ice flow model for mountain glaciers and its benchmark experiments},
  author={Wang, Yuzhe and Zhang, Tong and Xiao, Cunde and Ren, Jiawen and Wang, Yanfen},
  journal={Computers \& geosciences},
  volume={141},
  pages={104526},
  year={2020},
  publisher={Elsevier}
}



