from Evaluation import evaluation
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
import cv2 as cv


def Model_CAutoEncoder(train_data, train_target, test_data, test_target):
    IMG_SIZE = [28, 28, 1]
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = np.resize(train_data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Feat1 = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = np.resize(test_data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Feat2 = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    input_img = Input(shape=(28, 28, 1))

    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()

    # Extract features using the encoder part of the autoencoder
    encoder = Model(input_img, x)
    encoded_imgs = encoder.predict(Feat1)
    encoded_imgs_test = encoder.predict(Feat2)

    # Flatten the encoded images for classification
    encoded_imgs = encoded_imgs.reshape((encoded_imgs.shape[0], -1))
    encoded_imgs_test = encoded_imgs_test.reshape((encoded_imgs_test.shape[0], -1))

    # Build the classifier model
    classifier = Sequential()
    classifier.add(Dense(128, activation='relu', input_shape=(encoded_imgs.shape[1],)))
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dense(1, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()

    # Train the classifier
    classifier.fit(encoded_imgs, train_target, epochs=150)

    pred = classifier.predict(encoded_imgs_test)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = np.resize(pred, (120, 8))
    Eval = evaluation(pred, test_target)
    return Eval, pred