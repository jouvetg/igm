from .cnns import CNN
from .mlps import MLP
from .nos import FNO
from .dahunet import DahuNet
from .hybrid_dahunet_fno import HybridDahuNetFNO

Architectures = {
    "cnn":     CNN,
    "mlp":     MLP,
    "fno":     FNO,
    "dahunet": DahuNet,
    "hybrid_dahunet_fno": HybridDahuNetFNO,
}
