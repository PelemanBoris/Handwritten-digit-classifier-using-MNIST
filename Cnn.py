import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
(train_img, train_lab), (test_img, test_lab) = mnist.load_data()
train_img = train_img.reshape(60000,28,28,1)
test_img = test_img.reshape(10000,28,28,1)
train_img = keras.utils.normalize(train_img, axis=1)
test_img = keras.utils.normalize(test_img, axis =1)
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(3,3))
model.add(Conv2D(16, (3,3)))
model.add(MaxPooling2D(3,3))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(10, activation="softmax"))
print(model.summary())
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
metrics=['accuracy'])
model.fit(train_img, train_lab, epochs=10)
print(model.evaluate(test_img, test_lab))
model.save('model.h5')
