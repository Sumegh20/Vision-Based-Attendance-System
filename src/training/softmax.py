import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

class SoftMax():
    def __init__(self, input_shape, class_number):
        self.input_shape = input_shape
        self.class_number = class_number

    def modelBuild(self):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=self.input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.class_number, activation='softmax'))

        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model