from keras.datasets import cifar10,mnist
import numpy as np
def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    from keras.utils import to_categorical
    x_train=x_train/255.0
    y_train=to_categorical(y_train,10)
    x_test=x_test/255.0
    y_test=to_categorical(y_test,10)
    print("Cifar10: ",x_train.shape,y_train.shape,[np.max(x_train),np.min(x_train)])
    return x_train,y_train,x_test,y_test