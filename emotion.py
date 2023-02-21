from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Activation, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.image import resize
import numpy as np

class Emotion():

    def __init__(self):
        self.img_width = 48  # Ширина изображения
        self.img_height = 48  # Высота изображения
        self.model = None
        self.classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def fit(self):
        input_shape = (self.img_width, self.img_height, 1)

        self.model = Sequential()

        self.model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))

        self.model.add(Flatten())
        self.model.add(Dense(len(self.classes), activation='softmax'))

        self.model.load_weights('emotion_model_weights.h5')
        # return model

    def predict(self, image):
        face_image = resize(image, (self.img_width, self.img_height))
        face_image = np.expand_dims(face_image, axis=0)
        emotion_index = np.argmax(self.model.predict(face_image))
        emotion = self.classes[emotion_index]
        return emotion
