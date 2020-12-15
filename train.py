import tensorflow as tf
import data_loading
from models.zoomnn import ZoomNN
from models.unet import UNet
import datetime

if __name__ == '__main__':
    train_data = data_loading.build_dataset('train')
    # val_data = data_loading.build_dataset('val')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time

    model = UNet()

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0)]

    print('Compiling...')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss,
        metrics=metrics,
    )

    cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        update_freq=1000
    )

    print('Training...')
    model.fit(train_data,
        epochs=10,
        callbacks=[cb]
    )
