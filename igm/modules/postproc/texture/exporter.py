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
        n_dimensions = len(padding_parameters)
        for i, dimension in enumerate(padding_parameters):
            if dimension != (0, 0):
                begin = [0] * n_dimensions
                size = [-1] * n_dimensions
                begin[i] = dimension[0]
                size[i] = image.shape[i] - dimension[0] - dimension[1]
                unpadded_image = tf.slice(image, begin=begin, size=size)

        return unpadded_image

    def export(self, image: tf.Tensor, filename: str) -> None:
        fig = plt.figure()
        plt.imshow(image, origin="lower")
        plt.axis("off")
        plt.savefig(
            filename,
            bbox_inches="tight",
            # dpi=300,
            pad_inches=0,
        )
        plt.close(fig)


class TiffExporter(Exporter):
    def export(self, image: tf.Tensor, filename: str):
        raise NotImplementedError(
            "Tiff/tif exporter not implemented yet. Choose (png) please."
        )
        # TODO: export image as tiff