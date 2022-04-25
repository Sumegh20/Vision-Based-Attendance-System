from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold

import numpy as np
import pickle

from src.training.softmax import SoftMax


class TrainFaceRecognitionModel:
    def __init__(self, args):
        try:
            self.args = args
            file = open(self.args["embeddings"], "rb")
            self.data = pickle.load(file)
        except Exception as e:
            raise Exception(f"(TrainFaceRecognitionModel(__init__)): Something went wrong on initialising function \n" + str(e))

    def trainKerasModelForFaceRecognition(self):
        try:
            # preprocessing the data
            le = LabelEncoder()
            labels = le.fit_transform(self.data["names"])
            class_number = len(np.unique(labels))
            labels = labels.reshape(-1, 1)
            one_hot_encoder = OneHotEncoder(categorical_features=[0])

            labels = one_hot_encoder.fit_transform(labels).toarray()  # Target column
            embeddings = np.array(self.data["embeddings"])            # independent columns

            # Initialize Softmax training model
            BATCH_SIZE = 8
            EPOCHS = 5
            input_shape = embeddings.shape[1]

            # Build softmax classifier
            softmax = SoftMax(input_shape=(input_shape,), class_number=class_number)
            model = softmax.modelBuild()

            # Kfold
            cv = KFold(n_splits=5, random_state=42, shuffle=True)
            history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

            # Train
            for train, valid in cv.split(embeddings):
                X_train, X_val, y_train, y_val = embeddings[train], embeddings[valid], labels[train], labels[valid]
                his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
                print(his.history['acc'])

                history['acc'] += his.history['acc']
                history['val_acc'] += his.history['val_acc']
                history['loss'] += his.history['loss']
                history['val_loss'] += his.history['val_loss']

            # write the face recognition model to output
            model.save(self.args['model'])
            f = open(self.args["labelEncoder"], "wb")
            f.write(pickle.dumps(le))
            f.close()
        except Exception as e:
            raise Exception(f"(trainKerasModelForFaceRecognition): Something went wrong in trainKerasModelForFaceRecognition\n" + str(e))
