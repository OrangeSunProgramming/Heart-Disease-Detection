import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Layer, MaxPooling2D, Flatten, Conv2D, Activation, LSTM, Reshape, DepthwiseConv1D, DepthwiseConv2D, LSTM, Bidirectional, concatenate, Conv2DTranspose
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential


#Since we are using spectrograms, we are going to build a CNN model that effectively captures low and high features.
# The CNN seems effective and when stacking more layers deep in the architecutre. Stacking CNN towards the beginning blocks seemed effective in
# learning the low level feautures like edges, corners, and so on, but not stacking enough towards the last blocks makes the model not effective enough for our case.
# Therefore, the best way to handle this situation is to stack some CNNs towards the beginning blocks and stacking even more CNNs from the middle towards the end blocks.
# This way the model seems effective in being able to capture low and high level features.


def build_cnn(input_shape, conv1, conv2, conv3, conv4, conv5, drop1, drop2, dense1, dense2, dense3, dense4):
    inputs = Input(shape=input_shape)
    x = Conv2D(conv1, kernel_size=3, padding='same', activation='relu')(inputs)
    x = Conv2D(conv1, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(conv2, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv2, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(conv3, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv3, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv3, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(conv4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv4, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv4, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dropout(drop1)(x)
    
    x = Conv2D(conv5, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv5, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv5, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(conv5, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(dense1, activation='relu')(x)
    x = Dense(dense2, activation='relu')(x)
    x = Dropout(drop2)(x)
    x = Dense(dense3, activation='relu')(x)
    x = Dense(dense4, activation='relu')(x)
    outputs = Dense(5, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


def build_unet(input_shape, num_labels):
    inputs = Input(shape=input_shape)

    # Encoder: Convolutional Blocks
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder: Upsampling and concatenation
    u6 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output layer for multi-label classification
    outputs = Conv2D(num_labels, (1, 1), activation='sigmoid')(c9)

    # Create the model
    model = Model(inputs=[inputs], outputs=[outputs])

    return model