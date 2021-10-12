import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPool2D,BatchNormalization,\
    AveragePooling2D,Dropout,Flatten,Dense,Concatenate
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model


with open('dataset/training_data_binary_bit_image','rb') as file:
    x_train=np.load(file)
    y_train=np.load(file)

    x_val=np.load(file)
    y_val=np.load(file)

    x_test=np.load(file)
    y_test=np.load(file)


batch_size=32
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(batch_size)



def stem(input):

    c1=Conv2D(32,kernel_size=(3,3),activation='relu',padding='same')(input)
    c2=Conv2D(32,kernel_size=(3,3),activation='relu')(c1)
    m1=MaxPool2D(pool_size=(3,3),strides=(2,2))(c2)
    c4=Conv2D(64,kernel_size=(1,1),activation='relu',padding='same')(m1)
    c5=Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(c4)
    c6=Conv2D(128,kernel_size=(3,3),activation='relu',padding='same')(c5)
    b=relu(c6)

    return b


def inception_resnet_A(input):

    ir1=Conv2D(32,kernel_size=(1,1),activation='relu',padding='same')(input)

    ir2=Conv2D(32,kernel_size=(1,1),activation='relu',padding='same')(input)
    ir2=Conv2D(32,kernel_size=(3,3),activation='relu',padding='same')(ir2)

    ir3=Conv2D(32, kernel_size=(1,1),activation='relu',padding='same')(input)
    ir3 = Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same')(ir3)
    ir3 = Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same')(ir3)

    ir_merge=Concatenate()([ir1,ir2,ir3])

    ir_conv=Conv2D(128,kernel_size=(1,1),activation='linear',padding='same')(ir_merge)

    out=input+ir_conv
    out = relu(out)

    return out


def reduction_A(input,k=96, l=96, m=128, n=192):

    r1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(input)

    r2 = Conv2D(n, kernel_size=(3, 3), strides=(2,2), activation='relu')(input)

    r3 = Conv2D(k, kernel_size=(1, 1), activation='relu',padding='same')(input)
    r3 = Conv2D(l, kernel_size=(3, 3), activation='relu',padding='same')(r3)
    r3 = Conv2D(m, kernel_size=(3, 3), activation='relu', strides=(2,2),padding='valid')(r3)

    m = Concatenate()([r1, r2, r3])
    m = relu(m)
    return m


def inception_resnet_B(input):


    ir1 = Conv2D(64,kernel_size=(1, 1), activation='relu', padding='same')(input)

    ir2 = Conv2D(64, kernel_size=(1, 1), activation='relu', padding='same')(input)
    ir2 = Conv2D(64, kernel_size=(1, 3), activation='relu', padding='same')(ir2)
    ir2 = Conv2D(64, kernel_size=(3, 1), activation='relu', padding='same')(ir2)

    ir_merge = Concatenate()([ir1, ir2])

    ir_conv = Conv2D(448, kernel_size=(1, 1), activation='linear',padding='same')(ir_merge)

    out = input + ir_conv
    out = relu(out)
    return out


def reduction_B(input):

    r1 = MaxPool2D(pool_size=(3,3), strides=(2,2))(input)

    r2 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same')(input)
    r2 = Conv2D(192, kernel_size=(3, 3), activation='relu', strides=(2,2))(r2)

    r3 = Conv2D(128, kernel_size=(1, 1), activation='relu',padding='same')(input)
    r3 = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(2, 2))(r3)

    r4 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same')(input)
    r4 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(r4)
    r4 = Conv2D(128, kernel_size=(3, 3), activation='relu', strides=(2, 2))(r4)

    m = Concatenate()([r1, r2, r3, r4])
    m = relu(m)
    return m



# Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
input=tf.keras.layers.Input((29,29,1))
x = stem(input)
x = inception_resnet_A(x)
x = reduction_A(x, k=96, l=96, m=128, n=192)
x = inception_resnet_B(x)
x = reduction_B(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.8)(x)
x = Flatten()(x)
out = Dense(2, activation='softmax')(x)

model = Model(inputs=input, outputs=out)
learn_r=0.001
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learn_r),
              loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)
model.fit(train_data, epochs=100, validation_data=(x_val,y_val), callbacks=[early_stop])

model.save('saved_model/kds_conv_model')
