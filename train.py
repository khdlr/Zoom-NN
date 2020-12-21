import tensorflow as tf
import data_loading
from models.zoomnn import ZoomNN
from models.unet import UNet
import datetime


def ignore_m1_bce(y_true, y_pred):
    B = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (B, -1))
    y_pred = tf.reshape(y_pred, (B, -1))
    valid_value_mask = tf.math.not_equal(y_true, tf.constant(-1, tf.float32))

    y_true = tf.cast(y_true, tf.float32)
    bce = -tf.reduce_sum(tf.where(valid_value_mask,
        # Then
        y_true * tf.math.log_sigmoid(y_pred) +
        (1 - y_true) * tf.math.log_sigmoid(-y_pred),
        # Else
        tf.constant(0, tf.float32)
    ), axis=1)
    valid = tf.reduce_sum(tf.cast(valid_value_mask, tf.float32), axis=1)

    return bce / valid


def ignore_m1_cce(y_true, y_pred):
    y_true = tf.squeeze(y_true, 3)
    valid_value_mask = tf.math.not_equal(y_true, tf.constant(-1, tf.int8))
    y_true = tf.nn.relu(y_true)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        tf.cast(y_true, tf.int32), y_pred
    )
    loss = tf.where(valid_value_mask, loss, tf.constant(0.0, tf.float32))
    return tf.reduce_mean(loss, axis=[1, 2])


class BinaryAccuracyWithIgnore(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        valid = y_true != -1
        y_true = y_true > self.threshold
        y_pred = tf.nn.sigmoid(y_pred) > self.threshold
        correct = tf.math.logical_and(tf.equal(y_true, y_pred), valid)

        valid = tf.cast(valid, self.dtype)
        correct = tf.cast(correct, self.dtype)
        self.total.assign_add(tf.reduce_sum(valid))
        self.correct.assign_add(tf.reduce_sum(correct))

    def result(self):
        return self.correct / self.total 

    def reset_states(self):
        self.correct.assign(0)
        self.total.assign(0)




if __name__ == '__main__':
    model = ZoomNN()
    train_data = data_loading.build_dataset('train', 'zoomnn')
    val_data = data_loading.build_dataset('val', 'zoomnn')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time

    # loss = ignore_m1_bce
    loss = ignore_m1_bce
    metrics = [
        BinaryAccuracyWithIgnore(threshold=0.1, name='acc@0.1'),
        BinaryAccuracyWithIgnore(threshold=0.3, name='acc@0.3'),
        BinaryAccuracyWithIgnore(threshold=0.5, name='acc@0.5'),
    ]

    print('Compiling...')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics,
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq=1000
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + '/model.{epoch:02d}.{val_accuracy:03f}.h5',
    )

    print('Training...')
    model.fit(train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[ckpt, tensorboard]
    )
