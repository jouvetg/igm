import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from .normalizer import Normalizer

class Exporter(ABC):
    @abstractmethod
    def export(self, image: tf.Tensor, filename: str) -> None:
        pass


class PngExporter(Exporter):
    def convert_8bit(self, image: tf.Tensor, image_normalizer: Normalizer) -> tf.Tensor:
        out = image_normalizer.unnormalize(image)
        out = tf.cast(out, tf.uint8)
        out = tf.squeeze(out).numpy()

        return out

    def unpad(self, image: tf.Tensor, padding_parameters) -> tf.Tensor:
        import numpy as np
        n_dimensions = len(padding_parameters)
        padded_image = image
        for i, dimension in enumerate(padding_parameters):
            if not np.all(np.equal(dimension,0)):
                begin = [0] * n_dimensions
                size = [-1] * n_dimensions
                begin[i] = dimension[0]
                size[i] = image.shape[i] - dimension[0] - dimension[1]
                padded_image = tf.slice(padded_image, begin=begin, size=size)
        unpadded_image = padded_image
        
        return unpadded_image

    def export(self, image: tf.Tensor, filename: str) -> None:
        plt.imsave(fname=filename, arr=image, format='png', dpi=300, origin='lower')


class TiffExporter(Exporter):
    def export(self, image: tf.Tensor, filename: str):
        raise NotImplementedError(
            "Tiff/tif exporter not implemented yet. Choose (png) please."
        )
        # TODO: export image as tiff