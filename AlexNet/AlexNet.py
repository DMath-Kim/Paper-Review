import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

val_x, val_y = train_x[:5000], train_y[:5000]
train_x, train_y = train_x[5000:], train_y[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))

# Resizing of the images from 32x32 to 227x227

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 227x227
    image = tf.image.resize(image, (227, 227))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()

print("Training data size: ", train_ds_size)
print("Test data size: ", test_ds_size)
print("Validation data size: ", val_ds_size)

# process_images 를 이용하여 전처리를 하여 리사이즈 해준 이미지를 buffer_size 에 맞게 가지고와서 그 중 batch_size 만큼
# pick 하는 방법
train_ds = (train_ds.map(process_images)
                    .shuffle(buffer_size=train_ds_size)
                    .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds.map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
val_ds = (val_ds.map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32, drop_remainder=True))

alexnet = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=(227,227,3)),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu', padding='same'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(4096, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(4096, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(10, activation='softmax')
])

# TensorBoard is a tool that provides a suite of visualization and monitoring mechanisms. For the work in this tutorial,
# we’ll be utilizing TensorBoard to monitor the progress of the training of the network.

root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

alexnet.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics='acc')
alexnet.summary()
alexnet.fit(train_ds, epochs=20, validation_data=val_ds, validation_freq=1, callbacks=[tensorboard_cb])
alexnet.evaluate(test_ds)

tensorboard --logdir logs
