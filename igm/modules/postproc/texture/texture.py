from tensorflow.keras.models import load_model
import tensorflow as tf

from typing import Any
import time
import os
import igm

from .normalizer import FeatureNormalizer, ImageNormalizer
from .utils import TextureModelNotFoundError, resize_image
from .emulator import LinearRegressor
from .exporter import PngExporter, TiffExporter
from .constants import FeatureConstants, ImageConstants
from .preparer import Pix2PixHDImagePreparer
from .image_data import ImageData
from .pix2pixhd import Pix2PixHDPipeline

TEXTURE_DEFAULT_DIR = igm.__path__[0] + "/modules/postproc/texture/"
TEXTURE_DEFAULT_PATH = os.path.join(TEXTURE_DEFAULT_DIR, "pix2pixhd-texture-model")
# print(TEXTURE_DEFAULT_PATH)
# import sys
# sys.exit()
def params(parser: Any) -> None:
    parser.add_argument(
        "--texture_format",
        type=str,
        default="png",
        help="Format of the texture image (png, tif, or tiff)",
    )
    parser.add_argument(
        "--texture_model_path",
        type=str,
        default=TEXTURE_DEFAULT_PATH,
        help="Name of the folder for the texture model (tf format)",
    )
    parser.add_argument(
        "--resize_texture_resolution",
        type=int,
        default=1024,
        help="Resolution for the long-edge of the model (fixed at 1024 now for compatibility with the model)",
    )
    
    parser.add_argument(
        "--divide_by_density",
        type=float,
        default=1,
        help="This parameter solves an imcompatibility (this option will be removed in the future)",
    )
    # TODO: Add an option where they can overwrite certain features with their own data (e.g. ndvi, water, etc.), so one can you high resolution topg for example...
    # TODO: Add a logger for verbosity

def initialize(params: Any, state: Any) -> None:
    state.tcomp_texture = []
    if not os.path.exists(params.texture_model_path):
        model_url = "https://drive.google.com/drive/folders/1UP761XQpD4UvqNtbKNO20EXrxmAd4uoA?usp=sharing"

        # TODO: Only available to download folder if you use external packages (i do not know, but for now, I will let the user manually download from google drive)

        raise TextureModelNotFoundError(
            f"Model not found.\n\nPlease download the model\n{model_url})\nand place the downloaded folder in the following directory:\n{TEXTURE_DEFAULT_DIR}"
        )

    state.texture_model = load_model(params.texture_model_path, compile=False)

    feature_constants = FeatureConstants()
    image_constants = ImageConstants()
    state.feature_normalizer = FeatureNormalizer(constants=feature_constants)
    state.image_normalizer = ImageNormalizer(constants=image_constants)

    b = 15.335304
    coefficients = tf.constant([2.0448825, 0.62188774], shape=(1, 2))
    state.ndvi_emulator = LinearRegressor(coefficients=coefficients, b=b)

    if params.texture_format == "png":
        state.texture_exporter = PngExporter()
    elif params.texture_format == ("tif" or "tiff"):
        state.texture_exporter = TiffExporter()
    else:
        raise NotImplementedError(
            "Texture format not implemented. Please choose one of the following: (png, tif, or tiff)"
        )


def update(params: Any, state: Any) -> None:
    if state.saveresult:
        state.tcomp_texture.append(time.time())

        preparer = Pix2PixHDImagePreparer(
            state=state, params=params, ndvi_emulator=state.ndvi_emulator
        )
        preparer.get_features()
        image = preparer.prepare_batch()

        data = ImageData(values=image, height=image.shape[1], width=image.shape[2], state=state)
        
        new_width, new_height = data.compute_shape(
            resolution=params.resize_texture_resolution
        )
        data.upsample(width=new_width, height=new_height)

        if not data.square:
            padding_parameters = data.make_square()
            state.texture_exporter.padding_parameters = padding_parameters

        pipeline = Pix2PixHDPipeline(
            feature_normalizer=state.feature_normalizer,
            image_normalizer=state.image_normalizer,
            image=data,
            model=state.texture_model,
            exporter=state.texture_exporter,
            state=state,
            params=params,
        )

        pipeline.run()

        state.tcomp_texture[-1] -= time.time()
        state.tcomp_texture[-1] *= -1


def finalize(params: Any, state: Any) -> None:
    pass
