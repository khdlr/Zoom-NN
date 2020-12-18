import tensorflow as tf

from tensorflow.keras import Model, Input, Sequential, layers
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
    s4 = Input([256, 256, 2+14], name='input_s4')
    # s5 = Input([256, 256, 2+14], name='input_s5')

    C = base_channels

    # Initialize feature maps
    l1 = ConvReLU(C, batch_norm=False)(s1)
    l2 = ConvReLU(C, batch_norm=False)(s2)
    l3 = ConvReLU(C, batch_norm=False)(s3)
    l4 = ConvReLU(C, batch_norm=False)(s4)
    # l5 = ConvReLU(C, batch_norm=False)(s5)

    # Initial Dense Blocks
    x1 = ResidualDenseBlock(C)(l1)
    x2 = ResidualDenseBlock(C)(l2)
    x3 = ResidualDenseBlock(C)(l3)
    x4 = ResidualDenseBlock(C)(l4)
    # x5 = ResidualDenseBlock(C)(l5)

    # "Weaving" stage
    # x2 = layers.Add()([x2, DownAndPad()(x1)])
    # x2 = ResidualDenseBlock(C)(x2)

    # x3 = layers.Add()([x3, DownAndPad()(x2)])
    # x3 = ResidualDenseBlock(C)(x3)

    # x4 = layers.Add()([x4, DownAndPad()(x3)])
    # x4 = ResidualDenseBlock(C)(x4)

    # x5 = layers.Add()([x5, DownAndPad()(x4)])
    # x5 = ResidualDenseBlock(C)(x5)

    # x4 = layers.Add()([x4, CropAndUp()(x5)])
    # x4 = ResidualDenseBlock(C)(x4)

    x3 = layers.Add()([x3, CropAndUp()(x4)])
    x3 = ResidualDenseBlock(C)(x3)

    x2 = layers.Add()([x2, CropAndUp()(x3)])
    x2 = ResidualDenseBlock(C)(x2)

    x1 = layers.Add()([x1, CropAndUp()(x2)])
    x1 = ResidualDenseBlock(C)(x1)

    x1 = ResidualDenseBlock(C)(x1)

    out = layers.Conv2D(1, 1)(x1)

    return Model(inputs=[s1, s2, s3, s4], outputs=[out])


class ConvReLU(layers.Layer):
    def __init__(self, c_out, dilation=1, batch_norm=True):
        super().__init__()
        self.conv = layers.Conv2D(c_out, 3, dilation_rate=dilation, padding='same')
        self.c_out = c_out
        self.batch_norm = batch_norm
        self.dilation = dilation
        if batch_norm:
            self.bn = layers.BatchNormalization()
        self.relu = layers.PReLU()

    def get_config(self):
        return dict(c_out=self.c_out, dilation=self.dilation, batch_norm=self.batch_norm)

    def call(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualDenseBlock(layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bn = layers.BatchNormalization()
        self.cr1 = ConvReLU(8)
        self.cr2 = ConvReLU(8, dilation=3)
        self.cr3 = ConvReLU(8, dilation=3)
        self.cr4 = ConvReLU(8)

        self.final = layers.Conv2D(channels, 1)

        self.cat = layers.Concatenate()
        self.add = layers.Add()

    def get_config(self):
        return dict(channels=self.channels)

    def call(self, x):
        skip = x
        x = self.bn(x)
        x_new = self.cr1(x)
        x = self.cat([x, x_new])
        x_new = self.cr2(x)
        x = self.cat([x, x_new])
        x_new = self.cr3(x)
        x = self.cat([x, x_new])
        x_new = self.cr4(x)
        x = self.cat([x, x_new])

        x = self.add([skip, self.final(x)])
        return x


class CropAndUp(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        H = K.shape(x)[1]
        W = K.shape(x)[2]
        x = x[:,H//4:H-H//4,W//4:W-W//4]
        return K.resize_images(x, 2, 2, K.image_data_format(), interpolation='bilinear')


class DownAndPad(layers.Layer):
    def __init__(self):
        super().__init__()
        self.pool = layers.MaxPooling2D(2)

    def call(self, x):
        H = K.shape(x)[1]
        W = K.shape(x)[2]

        x = self.pool(x)
        x = tf.pad(x, ((0, 0), (H // 4, H // 4), (W // 4, W // 4), (0, 0)))
        return x
