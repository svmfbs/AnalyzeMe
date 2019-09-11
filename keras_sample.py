""" This is keras sample program """
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
sys.dont_write_bytecode = True # dont make .pyc files

# ####################################
# First example: Two-Layer Network
# ####################################
def F(x1, x2):
    """ test function """
    return np.sin(np.pi*x1/2.0)*np.cos(np.pi*x2/4.0)

A = 2
nb_samples = 1000
X_train = np.random.uniform(-A, +A, (nb_samples, 2))
Y_train = np.vectorize(F)(X_train[:, 0], X_train[:, 1])

model = Sequential()

nb_neurons = 20
model.add(Dense(nb_neurons, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dense(1))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(X_train, Y_train, epochs=10, batch_size=32)

x = [1.5, 0.5]
print(F(x[0], x[1]))

x = np.array(x).reshape(1, 2)
print(x)
print(model.predict(x))
print(model.predict(x)[0][0])

Width = 200
Height = 200
U = np.linspace(-A, A, Width)
V = np.linspace(-A, A, Height)
UV = np.transpose([np.tile(U, len(V)), np.tile(V, len(U))])
print(UV)
ys = model.predict(UV)
print(ys)
l = ys.reshape(Width, Height)
print(l)

# ####################################
# Second example: CNN for MNIST
# ####################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_train /= 255
print(x_train.shape)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255

print(y_train.shape)
print(y_train[0:3])

y_train = np_utils.to_categorical(y_train, 10)
print(y_train[0])

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
print(model.output_shape)

model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)

model.add(Dropout(0.25))
print(model.output_shape)

model.add(Flatten())
print(model.output_shape)

model.add(Dense(128, activation='relu'))
print(model.output_shape)

model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.output_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
# print(model.predict(x_test[0].reshape(1, 28, 28, 1)))

# ####################################
# Third example: Using VGG to recognize objects
# ####################################
model = VGG16()
print(model.summary())
