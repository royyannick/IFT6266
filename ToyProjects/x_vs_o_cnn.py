'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import random

x_train = data_x
y_train = data_y
x_test = data_x
y_test = data_y

batch_size = 128
num_classes = 2
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32

x_train = x_train.astype('float32')
x_train = x_train.reshape(len(x_train), img_rows,img_cols,1)
x_test = x_test.astype('float32')
x_test = x_test.reshape(len(x_test), img_rows,img_cols,1)
x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



num_pred = random.randint(1,len(x_test))

pred = model.predict(x_test[num_pred,].reshape(1,32,32,1))

plt.figure()
plt.imshow(x_test[num_pred,].reshape(32,32))
plt.axis("off")
if pred[0][0] == 1:
    plt.title("circle")
else:
    plt.title("cross")