import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import cv2
import os


# Saving segmented images
def save_segmented_images(model, images):
    num_classes = 10
    predictions = model.predict(images)
    predictions = np.argmax(predictions, axis=-1)

    for i, prediction in enumerate(predictions):
        colormap = plt.get_cmap('jet')
        colored_image = colormap(prediction / (num_classes - 1))
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
        return colored_image


# Synthetic dataset for demonstration
def create_synthetic_data(num_samples, img_shape, num_classes):
    X = np.random.rand(num_samples, *img_shape)
    y = np.random.randint(0, num_classes, (num_samples, img_shape[0], img_shape[1]))
    return X, y


# Custom layers
class ResidualConvUnit(layers.Layer):
    def __init__(self, filters):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.relu = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, padding='same')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.conv2(x)
        x = layers.add([x, inputs])
        x = self.relu(x)
        return x


class MultiResolutionFusion(layers.Layer):
    def __init__(self, filters):
        super(MultiResolutionFusion, self).__init__()
        self.conv_low = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.conv_high = layers.Conv2D(filters, kernel_size=3, padding='same')

    def call(self, low_res_input, high_res_input):
        low_res_input = layers.UpSampling2D()(low_res_input)
        low_res_input = self.conv_low(low_res_input)
        high_res_input = self.conv_high(high_res_input)
        return layers.add([low_res_input, high_res_input])


class ChainedResidualPooling(layers.Layer):
    def __init__(self, filters):
        super(ChainedResidualPooling, self).__init__()
        self.pool = layers.MaxPooling2D(pool_size=5, strides=1, padding='same')
        self.conv = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.relu = layers.ReLU()

    def call(self, inputs):
        x = self.relu(inputs)
        x = self.pool(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.conv(x)
        return x


class RefineNetBlock(layers.Layer):
    def __init__(self, filters):
        super(RefineNetBlock, self).__init__()
        self.rcu_low = ResidualConvUnit(filters)
        self.rcu_high = ResidualConvUnit(filters)
        self.mrf = MultiResolutionFusion(filters)
        self.crp = ChainedResidualPooling(filters)
        self.rcu_output = ResidualConvUnit(filters)

    def call(self, low_res_input, high_res_input):
        low_res_input = self.rcu_low(low_res_input)
        high_res_input = self.rcu_high(high_res_input)
        x = self.mrf(low_res_input, high_res_input)
        x = self.crp(x)
        x = self.rcu_output(x)
        return x


def build_refinenet(input_shape, num_classes, sol):
    inputs = layers.Input(shape=input_shape)

    # Example backbone network (simple version of ResNet)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Example feature extraction layers
    c1 = layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    c2 = layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(c1)
    c3 = layers.Conv2D(1024, kernel_size=3, strides=2, padding='same')(c2)
    c4 = layers.Conv2D(2048, kernel_size=3, strides=2, padding='same')(c3)

    # RefineNet blocks
    r3 = RefineNetBlock(sol[0])(c3, c4)
    r2 = RefineNetBlock(256)(c2, r3)
    r1 = RefineNetBlock(256)(c1, r2)

    # Upsample and output layer
    outputs = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')(r1)
    outputs = layers.UpSampling2D(size=(8, 8))(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def Model_ARNeT(Image, sol=None):
    if sol is None:
        sol = [5, 5, 300]
    IMG_SIZE = 224
    input_shape = (224, 224, 3)
    num_classes = 21
    Train_X = np.zeros((Image.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Image.shape[0]):
        temp = np.resize(Image[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    Test_X = np.zeros((Image.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Image.shape[0]):
        temp = np.resize(Image[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    model = build_refinenet(input_shape, num_classes, sol)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train, y_train = create_synthetic_data(10, input_shape, num_classes)
    X_val, y_val = create_synthetic_data(2, input_shape, num_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=sol[1], steps_per_epoch=sol[2])
    images = save_segmented_images(model, X_val)
    return images
