import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras import layers
import tensorflow.keras.backend as K

DEFAULTS = dict(
    base_channels=16,
    output_channels=2,
    batch_norm=False,
    stack_height=6,
)

def ZoomNN(config={}):
    config = {**DEFAULTS, **config}  # config overwrites defaults
    base_channels   = config['base_channels']
    output_channels = config['output_channels']
    batch_norm      = config['batch_norm']
    stack_height    = config['stack_height']

    s1 = Input([256, 256, 2], name='input_s1')
    s2 = Input([256, 256, 2], name='input_s2')
    s3 = Input([256, 256, 2], name='input_s3')
    s4 = Input([256, 256, 2], name='input_s4')
    s5 = Input([256, 256, 2], name='input_s5')

    b4 = Input([256, 256, 14], name='input_b4')
    b5 = Input([256, 256, 14], name='input_b5')

    # Initialize feature maps
    l1 = convrelu(s1, 32, batch_norm=False)
    l2 = convrelu(s2, 32, batch_norm=False)
    l3 = convrelu(s3, 32, batch_norm=False)
    l4 = layers.Add()([
        convrelu(s4, 32, batch_norm=False),
        convrelu(b4, 32, batch_norm=False)
    ])
    l5 = layers.Add()([
        convrelu(s5, 32, batch_norm=False),
        convrelu(b5, 32, batch_norm=False)
    ])

    # Initial Dense Blocks
    x1 = ResidualDenseBlock(l1)
    x2 = ResidualDenseBlock(l2)
    x3 = ResidualDenseBlock(l3)
    x4 = ResidualDenseBlock(l4)
    x5 = ResidualDenseBlock(l5)

    # "Weaving" stage
    x4 = layers.Add()([x4, CropAndUp()(x5)])
    x4 = ResidualDenseBlock(x4)

    x3 = layers.Add()([x3, CropAndUp()(x4)])
    x3 = ResidualDenseBlock(x3)

    x2 = layers.Add()([x2, CropAndUp()(x3)])
    x2 = ResidualDenseBlock(x2)

    x1 = layers.Add()([x1, CropAndUp()(x2)])
    x1 = ResidualDenseBlock(x1)

    out = layers.Conv2D(1, 1)(x1)

    return Model(inputs=[s1, s2, s3, s4, s5, b4, b5], outputs=[out])


def convrelu(x, c_out, batch_norm=True):
    x = layers.Conv2D(c_out, 3, padding='same')(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def ResidualDenseBlock(x):
    c_in = x.shape[-1]
    skip = x
    x = layers.BatchNormalization()(x)
    x_new = convrelu(x, 8)
    x = layers.Concatenate()([x, x_new])
    x_new = convrelu(x, 8)
    x = layers.Concatenate()([x, x_new])
    x_new = convrelu(x, 8)
    x = layers.Concatenate()([x, x_new])
    x_new = convrelu(x, 8)
    x = layers.Concatenate()([x, x_new])

    x = layers.Conv2D(c_in, 1)(x)
    return layers.Add()([skip, x])


class CropAndUp(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        H = K.shape(x)[1]
        W = K.shape(x)[2]
        x = x[:,H//4:H-H//4,W//4:W-W//4]
        return K.resize_images(x, 2, 2, K.image_data_format(), interpolation='bilinear')
