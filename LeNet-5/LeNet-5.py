import tensorflow as tf

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = train_x / 255.0
test_x = test_x / 255.0

train_y = tf.keras.utils.to_categorical(train_y)
test_y = tf.keras.utils.to_categorical(test_y)

train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)

val_x = train_x[:5000]
val_y = train_y[:5000]
train_x = train_x[5000:]
train_y = train_y[5000:]

lenet_5_model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(6, kernel_size=5, strides=1, activation='tanh', input_shape=train_x[0].shape, padding='same'),
tf.keras.layers.AveragePooling2D(),
tf.keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),
tf.keras.layers.AveragePooling2D(),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(120, activation='tanh'),
tf.keras.layers.Dense(84, activation='tanh'),
tf.keras.layers.Dense(10, activation='softmax')
])

lenet_5_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')
lenet_5_model.summary()
lenet_5_model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y))
history = lenet_5_model.evaluate(test_x, test_y)

print("Loss : ", history[0])
print("Accuracy : ", history[1])
