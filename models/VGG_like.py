from keras.models import Sequential
from keras import regularizers
from keras.layers import *
from quantize_layers import Conv2D_Q,Dense_Q,DepthwiseConv2D_Q
def VGG7_N(using_bn=True,
           weights_decay=0.0001,
           kq=None,
           bq=None,
           aq=None,
           after_activation=None,
           N=64,
           c_drop_rate=0,
           f_drop_rate=0.5):
    VGG7 = Sequential()
    #2 conv
    VGG7.add(Conv2D_Q(filters=N, kernel_size=(3, 3), activation='relu', input_shape=(32, 32,3),padding='same',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    if using_bn:
        VGG7.add(BatchNormalization())
    VGG7.add(Conv2D_Q(filters=N, kernel_size=(3, 3), activation='relu',padding='same',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    if using_bn:
        VGG7.add(BatchNormalization())
    VGG7.add(MaxPooling2D(pool_size=(2, 2)))
    VGG7.add(Dropout(c_drop_rate))
    #2 conv
    VGG7.add(Conv2D_Q(filters=2*N, kernel_size=(3, 3), activation='relu',padding='same',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    if using_bn:
        VGG7.add(BatchNormalization())
    VGG7.add(Conv2D_Q(filters=2*N, kernel_size=(3, 3), activation='relu',padding='same',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    if using_bn:
        VGG7.add(BatchNormalization())
    VGG7.add(MaxPooling2D(pool_size=(2, 2)))
    VGG7.add(Dropout(c_drop_rate))
    #2 conv
    VGG7.add(Conv2D_Q(filters=4*N, kernel_size=(3, 3), activation='relu',padding='same',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    if using_bn:
        VGG7.add(BatchNormalization())
    VGG7.add(Conv2D_Q(filters=4*N, kernel_size=(3, 3), activation='relu',padding='same',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    if using_bn:
        VGG7.add(BatchNormalization())
    VGG7.add(MaxPooling2D(pool_size=(2, 2)))
    VGG7.add(Dropout(c_drop_rate))
    #FC
    VGG7.add(Flatten())
    VGG7.add(Dense_Q(units=1024,activation='relu',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    if using_bn:
        VGG7.add(BatchNormalization())
    VGG7.add(Dropout(f_drop_rate))
    VGG7.add(Dense_Q(units=10,activation='softmax',
                       kernel_regularizer=regularizers.l2(weights_decay),
                       bias_regularizer=regularizers.l2(weights_decay),
                       kq=kq,
                       bq=bq,
                       aq=aq,
                       after_activation=after_activation))
    return VGG7