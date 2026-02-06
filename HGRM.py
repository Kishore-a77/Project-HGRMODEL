import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = r'C:\Users\kisho\OneDrive\Documents\Python Scripts\mnist.npz'
with np.load(path) as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

num_classes = 10

model = models.Sequential([
    Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class QuietCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Step {epoch + 1} completed')

history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), verbose=0, callbacks=[QuietCallback()])

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {test_acc}')
