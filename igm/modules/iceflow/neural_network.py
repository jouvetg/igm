import numpy as np 
import tensorflow as tf 

def cnn(cfg, nb_inputs, nb_outputs):
    """
    Routine serve to build a convolutional neural network
    """

    inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

    conv = inputs

    if cfg.iceflow.iceflow.iflo_activation == "LeakyReLU":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = getattr(tf.keras.layers,cfg.iceflow.iceflow.iflo_activation)()      

    for i in range(int(cfg.iceflow.iceflow.iflo_nb_layers)):
        conv = tf.keras.layers.Conv2D(
            filters=cfg.iceflow.iceflow.iflo_nb_out_filter,
            kernel_size=(cfg.iceflow.iceflow.iflo_conv_ker_size, cfg.iceflow.iceflow.iflo_conv_ker_size),
            kernel_initializer=cfg.iceflow.iceflow.iflo_weight_initialization,
            padding="same",
        )(conv)

        conv = activation(conv)

        conv = tf.keras.layers.Dropout(cfg.iceflow.iceflow.iflo_dropout_rate)(conv)

    outputs = conv

    outputs = tf.keras.layers.Conv2D(
        filters=nb_outputs,
        kernel_size=(
            1,
            1,
        ),
        kernel_initializer=cfg.iceflow.iceflow.iflo_weight_initialization,
        activation=None,
    )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def unet(params, nb_inputs, nb_outputs):
    """
    Routine serve to define a UNET network from keras_unet_collection
    """

    from keras_unet_collection import models

    layers = np.arange(int(params.iflo_nb_blocks))

    number_of_filters = [
        params.iflo_nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
    ]

    return models.unet_2d(
        (None, None, nb_inputs),
        number_of_filters,
        n_labels=nb_outputs,
        stack_num_down=2,
        stack_num_up=2,
        activation=params.iflo_activation,
        output_activation=None,
        batch_norm=False,
        pool="max",
        unpool=False,
        name="unet",
    )
