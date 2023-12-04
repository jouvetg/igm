import tensorflow as tf

from typing import Tuple, Any
from .emulator import Emulator

class Pix2PixHDImagePreparer:
    def __init__(self, params: Any, state: Any, ndvi_emulator: Emulator = None) -> None:
        if ndvi_emulator is None and (not hasattr(state, "ndvi")):
            raise ValueError(
                "Emulator is required for NDVI if no NDVI is provided."
            )  # ? getting a type error with None here... could be a bug to look into
        self.state = state
        self.params = params
        self.ndvi_emulator = ndvi_emulator

    def get_features(self) -> None:
        topg = self.state.topg
        thk = self.state.thk
        vx = self.state.U[-1]
        vy = self.state.V[-1]

        print(dir(self.state))
        try:
            air_temp = self.state.air_temp
        except AttributeError:
            raise AttributeError("Air temperature is required for texture model. Please include it in an input file with the variable name 'air_temp'. \
                                Also, note that air_temp will be averaged to obtain the annually averaged air_temp for the spatial grid.")
        
        try:
            precipitation = self.state.precipitation
        except AttributeError:
            raise AttributeError("Precipitation is required for texture model. Please include it in an input file with the variable name 'precipitation'. \
                                Also, note that precipitation will be averaged to obtain the annually averaged precipitation for the spatial grid.")
            
        # if the user supplies monthly, quarerly, or seasonal data, average to get annual data
        if air_temp.shape[0] > 1:
            temp = tf.math.reduce_mean(air_temp, axis=0)
        else:
            temp = air_temp
        if precipitation.shape[0] > 1:
            prec = tf.math.reduce_mean(precipitation, axis=0)
        else:
            prec = precipitation

        prec = self.prec_units(prec)

        if hasattr(self.state, "ndvi"):
            ndvi = self.state.ndvi
        else:
            ndvi = self.compute_ndvi(temp, prec)

        if hasattr(self.state, "water"):
            water = self.state.water
        else:
            water = self.compute_water(topg.shape)

        self.image = tf.stack([topg, water, vx, thk, prec, ndvi, vy, temp], axis=0)

    def prec_units(self, prec: tf.Tensor) -> tf.Tensor:
        return (prec / self.params.divide_by_density) * (1000 / 12)  # (m / yr. -> mm / month)

    def compute_ndvi(self, temp: tf.Tensor, prec: tf.Tensor) -> tf.Tensor:
        temp_flat = tf.reshape(temp, [-1])
        prec_flat = tf.reshape(prec, [-1])
        climate = tf.stack((temp_flat, prec_flat))

        ndvi = self.ndvi_emulator.apply(climate)
        ndvi = tf.reshape(ndvi, [temp.shape[0], temp.shape[1]])
        ndvi = tf.clip_by_value(ndvi, 0, 255)

        return ndvi

    def compute_water(self, tensor_shape: Tuple[int, int]):
        return tf.zeros(shape=tensor_shape)  # assumes no water for now...

    def prepare_batch(self) -> tf.Tensor:
        image = tf.transpose(self.image, [1, 2, 0])
        image = tf.expand_dims(image, axis=0)

        return image