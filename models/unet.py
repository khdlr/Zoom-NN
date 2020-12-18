import tensorflow as tf


from tensorflow.keras import Model, Input
from tensorflow.keras import layers
import tensorflow.keras.backend as K

DEFAULTS = dict(
    base_channels=16,
    output_channels=2,
    batch_norm=False,
    stack_height=4,
)


def UNet(config={}):
    config = {**DEFAULTS, **config}  # config overwrites defaults
    base_channels   = config['base_channels']
    output_channels = config['output_channels']
    batch_norm      = config['batch_norm']
    stack_height    = config['stack_height']

    x = Input([256, 256, 2])
    inputs = [x]

    # Initialize feature maps
    x = DoubleConv(x, base_channels, batch_norm)

    # Build the encoder part
    channels = base_channels
    skip_connections = []
    for i in range(stack_height):
        skip_connections.append(x)
        channels = channels * 2
        x = DownBlock(x, channels, batch_norm=batch_norm)

    # Build the decoder part
    scale = 1
    for skip in reversed(skip_connections):
        channels = channels // 2
        scale = scale * 2
        x = UpBlock(x, skip, channels, batch_norm=batch_norm)

    x = DoubleConv(x, base_channels, batch_norm)
    x = layers.Conv2D(1, 1)(x)

    outputs = [x]

    return Model(inputs=inputs, outputs=outputs)


class AttentionMerging(layers.Layer):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return dict()

    @classmethod
    def from_config(cls, config):
        return cls()

    def call(self, predictions, attentions):
        P = tf.stack(predictions, axis=4)
        A = tf.stack(attentions, axis=4)
        A = K.softmax(A, axis=4)
        shp = tf.shape(A)
        P = tf.reshape(P, [-1, shp[4]])
        A = tf.reshape(A, [-1, shp[4]])
        attended = K.batch_dot(P, A, axes=[1, 1])
        attended = tf.reshape(attended, shp[:4])
        return attended


class ExtractBundledDEM(layers.Layer):
    """Only needed if DEM is provided at the same resolution as
    the imagery. This might be the case for inference."""
    def __init__(self):
        super().__init__()

    def call(self, x):
        hh_hv, dem, ratio = tf.split(x, [2, 1, 1], axis=-1)
        dem = tf.clip_by_value(dem, -50, 50)

        avg_pooled = tf.nn.avg_pool2d(dem, 16, strides=16, padding='VALID')
        max_pooled = tf.nn.max_pool2d(dem, 16, strides=16, padding='VALID')
        min_pooled = -tf.nn.max_pool2d(-dem, 16, strides=16, padding='VALID')

        dem = tf.concat([avg_pooled, max_pooled, min_pooled], axis=-1)
        s1 = tf.concat([hh_hv, ratio], axis=-1)
        return s1, dem

    def compute_output_shape(self, input_shape):
        assert input_shape[-1] == 4
        H, W, C = input_shape
        return (H, W, 3), (H // 16, W // 16, 3)


def BundledDEMWrapper(hed_unet):
    """
    Wraps a HED-UNet instance to handle wrapped DEMs
    """
    input_bundled = Input([None, None, 4])
    inputs = [input_bundled]
    x, dem = ExtractBundledDEM()(input_bundled)
    out = hed_unet([x, dem])
    return Model(inputs=[input_bundled], outputs=[out])


def DoubleConv(x, channels, batch_norm=True):
    use_bias = not batch_norm  # Bias + batch norm is redundant 

    # First conv
    x = layers.Conv2D(channels, 3, padding='same', use_bias=use_bias)(x)
    if batch_norm:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)

    # Second conv
    x = layers.Conv2D(channels, 3, padding='same', use_bias=use_bias)(x)
    if batch_norm:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)

    return x


def DownBlock(x, channels, conv_block=DoubleConv, batch_norm=True):
    """UNet Downsampling Block"""
    use_bias = not batch_norm
    in_channels = x.shape[-1]
    # Down-sampling part
    x = layers.Conv2D(in_channels, 2, strides=2, use_bias=use_bias)(x)
    if batch_norm:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    # Conv Block
    x = conv_block(x, channels, batch_norm=batch_norm)
    return x


def UpBlock(x, skip, channels, conv_block=DoubleConv, batch_norm=True):
    """UNet Upsampling Block"""
    use_bias = not batch_norm
    in_channels = x.shape[-1]
    # Up-sampling part
    x = layers.Conv2DTranspose(in_channels // 2, 2, strides=2, use_bias=use_bias)(x)
    if batch_norm:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    # Concatenate Skip Connection
    x = layers.Concatenate()([x, skip])
    # Conv Block
    x = conv_block(x, channels, batch_norm=batch_norm)
    return x


class Upsampling(layers.Layer):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def get_config(self):
        return dict(scale_factor=self.scale_factor)

    @classmethod
    def from_config(cls, config):
        return cls(config['scale_factor'])

    def call(self, x):
        return K.resize_images(x, self.scale_factor, self.scale_factor,
                K.image_data_format(), interpolation='bilinear')

