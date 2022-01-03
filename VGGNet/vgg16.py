import os
import time
import tensorflow as tf
import numpy as np

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

val_x, val_y = train_x[:5000], train_y[:5000]
train_x, train_y = train_x[5000:], train_y[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 227x227
    image = tf.image.resize(image, (224, 224))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()

train_ds = (train_ds.map(process_images)
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds.map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
val_ds = (val_ds.map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32, drop_remainder=True))

vgg = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(units=4096, activation='relu'),
tf.keras.layers.Dense(units=4096, activation='relu'),
tf.keras.layers.Dense(units=10, activation='softmax'),
])

root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

vgg.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics='acc')
vgg.summary()
vgg.fit(train_ds, epochs=20, validation_data=val_ds, validation_freq=1, callbacks=[tensorboard_cb])
vgg.evaluate(test_ds)
