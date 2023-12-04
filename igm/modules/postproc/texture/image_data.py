import tensorflow as tf

from typing import Tuple
from dataclasses import dataclass

from .utils import resize_image
from igm import State

@dataclass
class ImageData:
    values: tf.Tensor
    height: int
    width: int
    state: State
    upsampled: bool = False
    square: bool = False
 
    def upsample(self, height: int, width: int) -> None:
        self.values = resize_image(self.values, height, width)
        self.upsampled = True

    def compute_shape(self, resolution: int) -> Tuple[int, int]:
        self.upsample_resolution = resolution
        if self.height == self.width:
            self.height = resolution
            self.width = resolution
            self.square = True

            return resolution, resolution
        elif self.height < self.width:
            ratio = resolution / self.width
            new_width = int(tf.math.ceil(ratio * self.width))
            new_height = int(tf.math.ceil(ratio * self.height))

            if new_height % 2 != 0:
                new_height += 1

            self.height = new_height
            self.width = new_width

            return new_width, new_height
        else:
            ratio = resolution / self.height
            new_width = int(tf.math.ceil(ratio * self.width))
            new_height = int(tf.math.ceil(ratio * self.height))

            if new_width % 2 != 0:
                new_width += 1

            self.height = new_height
            self.width = new_width

            return new_width, new_height

    def make_square(self, method: str = "reflect"):
        if not self.upsampled:
            self.upsample(self.height, self.width)
            padding_parameters = [
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ]  # [bs, h, w, c]
        if self.height > self.width:
            pad_width = (self.height - self.width) // 2
            padding_parameters = [
                (0, 0),
                (0, 0),
                (pad_width, pad_width),
                (0, 0),
            ]  # [bs, h, w + padding, c]
        else:
            pad_width = (self.width - self.height) // 2
            padding_parameters = [
                (0, 0),
                (pad_width, pad_width),
                (0, 0),
                (0, 0),
            ]  # [bs, h + padding, w, c]

        self.values = tf.pad(self.values, padding_parameters, mode=method)

        return padding_parameters

    def run(self) -> None:
        new_width, new_height = self.compute_shape(resolution=self.upsample_resolution)
        self.upsample(width=new_width, height=new_height)
        if not self.square:
            padding_parameters = self.make_square()
            self.state.texture_exporter.padding_parameters = padding_parameters