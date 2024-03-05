# from tensorflow.keras.models import load_model
import tensorflow as tf

from typing import Any
import time
import os
import igm
import logging
import numpy as np

from .normalizer import FeatureNormalizer, ImageNormalizer
from .utils import TextureModelNotFoundError, resize_image
from .emulator import LinearRegressor
from .exporter import PngExporter, TiffExporter
from .constants import FeatureConstants, ImageConstants
from .preparer import Pix2PixHDImagePreparer
from .image_data import ImageData
from .pix2pixhd import Pix2PixHDPipeline
from .pix2pixhd_model_assets.generator import LocalEnhancer
from .pix2pixhd_model_assets.loading import load_model

TEXTURE_DEFAULT_DIR = igm.__path__[0] + "/modules/postproc/texture/"
TEXTURE_CKPT_DIR = os.path.join(TEXTURE_DEFAULT_DIR, "checkpoints")

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
        default=TEXTURE_CKPT_DIR,
        help="Name of the folder for the texture model (tf format)",
    )
    parser.add_argument(
        "--texture_verbosity",
        type=int,
        default=30,
        help="Python Logger verbosity level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)",
    )
    
    parser.add_argument(
        "--divide_by_density",
        type=float,
        default=1,
        help="This parameter solves an imcompatibility (this option will be removed in the future)",
    )
    parser.add_argument(
        "--texture_resolution",
        type=float,
        default=-1,
        help="This parameter solves an imcompatibility (this option will be removed in the future)",
    )
    # TODO: Add an option where they can overwrite certain features with their own data (e.g. ndvi, water, etc.), so one can you high resolution topg for example...
    # TODO: Add a logger for verbosity

def initialize(params: Any, state: Any) -> None:
    state.tcomp_texture = []
    logging.basicConfig(level=params.texture_verbosity)
    if not os.path.exists(params.texture_model_path):
        model_url = "https://drive.google.com/drive/folders/1Rmw_tCVplnjGfhjnZtVDae7djOmu5ZKP?usp=sharing"

        # TODO: Only available to download folder if you use external packages (i do not know, but for now, I will let the user manually download from google drive)

        raise TextureModelNotFoundError(
            f"Model not found.\n\nPlease download the model\n{model_url})\nand place the downloaded folder in the following directory:\n{TEXTURE_DEFAULT_DIR}"
        )

    state.texture_model = LocalEnhancer(input_nc=8, output_nc=3, ngf=32, n_downsample_global=4, n_blocks_global=9, n_local_enhancers=1, n_blocks_local=3)
    checkpoint_dict = {"generator": state.texture_model}
    checkpoint = tf.train.Checkpoint(**checkpoint_dict)

    __ = load_model(checkpoint, TEXTURE_CKPT_DIR)

    feature_constants = FeatureConstants()
    image_constants = ImageConstants()
    state.feature_normalizer = FeatureNormalizer(constants=feature_constants)
    state.image_normalizer = ImageNormalizer(constants=image_constants)

    b = 15.335304
    coefficients = tf.constant([2.0448825, 0.62188774], shape=(1, 2))
    state.ndvi_emulator = LinearRegressor(coefficients=coefficients, b=b)

    if params.texture_format == "png":
        state.texture_exporter = PngExporter()
    elif params.texture_format == "tif" or params.texture_format == "tiff":
        state.texture_exporter = TiffExporter(state=state)
    else:
        raise NotImplementedError(
            "Texture format not implemented. Please choose one of the following: (png, tif, or tiff)"
        )

def is_power_of_two(number):
    """Checks if a number is a power of two"""
    return (number & (number - 1)) == 0

def nearest_power_of_two(number, method="ceil"):
    """Finds nearest power of two"""
    import math
    if method == "ceil":
        return 2 ** math.ceil(math.log2(number))
    elif method == "floor":
        return 2 ** math.floor(math.log2(number))
    else:
        return 2 ** round(math.log2(number))

def update(params: Any, state: Any) -> None:
    if state.saveresult:
        state.tcomp_texture.append(time.time())

        preparer = Pix2PixHDImagePreparer(
            state=state, params=params, ndvi_emulator=state.ndvi_emulator
        )
        preparer.get_features()
        image = preparer.prepare_batch()

        logging.info(f"Input Image shape (before resizing): {image.shape}")
        data = ImageData(values=image, height=image.shape[1], width=image.shape[2], state=state)
        logging.debug(f"Input Image (before resizing): {data.values}")
        if max(data.height, data.width) < 1024: # optimal resolution for pix2pixhd
            resolution = 1024
        else:
            if data.height > data.width:
                if not is_power_of_two(data.height):
                    resolution = nearest_power_of_two(data.height)
            elif data.height < data.width:
                if not is_power_of_two(data.width):
                    resolution = nearest_power_of_two(data.width)
            else:
                resolution = data.height # or data.width but its always going to be a square here
        
        # ! Not tested yet but trying it out
        if params.texture_resolution != -1: # overrides the resolution
            resolution = params.texture_resolution
            
        logging.info(f"Long side resolution for resizing: {resolution}")

        padding_parameters = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
        # iteratively resize image until it is at least 1024x1024 (due to tf.pad maximal padding restraints)
        while (data.height < resolution) or (data.width < resolution): # only in < 1024 case
            resolution_height = 2 * data.height if (2 * data.height <= resolution) else resolution
            resolution_width = 2 * data.width if (2 * data.width <= resolution) else resolution
            logging.info(f"Resolution size (HxW) for padding: {resolution_height}x{resolution_width}")
            padding_parameters += data.pad_image(resolution_height=resolution_height, resolution_width=resolution_width)
            logging.info(f"Resolution size (HxW) after padding: {data.height}x{data.width}")
            logging.debug(f"Cumulative padding parameters: {padding_parameters}")

        logging.info(f"Input Image shape (after resizing): {data.values.shape}")
        logging.debug(f"Input Image (after resizing): {data.values}")
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
