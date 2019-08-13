import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D, Dense, Flatten

INPUT_SHAPE = (64,64,1)

def get_model():
    input_0 = Input(shape=(64,64,4))

    conv_0 = Conv2D(32, 3, strides=(1,1), padding='same', activation='relu')(input_0)
    conv_1 = Conv2D(32, 3, strides=(2,2), padding='same', activation='relu')(conv_0)
    maxpool_1 = MaxPool2D(padding='same')(conv_1)
    conv_2 = Conv2D(64, 3, padding='same', activation='relu')(maxpool_1)

    flatten_0 = Flatten()(conv_2)
    fc_0 = Dense(256)(flatten_0)
    fc_1 = Dense(2, activation='linear')(fc_0)

    model = tf.keras.models.Model(input_0, fc_1)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')

    return model
