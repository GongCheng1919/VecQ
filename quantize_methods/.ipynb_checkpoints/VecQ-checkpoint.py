import keras.backend as K
def VecQ(bits=2):
    lambdas=[1.0000, 0.9957, 0.5860, 0.3352, 0.1881, 0.1041, 0.0569, 0.0308]
    if bits>8:
        lam=6/(2**(bits)-1)
    else:
        lam=lambdas[bits-1]
    mmax=2**(bits-1)-1
    mmin=-2**(bits-1)
    # calculating means and standard derivation
    def dot(a,b):
        return K.sum(a*b)
    def quantize(w):
        std=K.std(w)
        tmp=w/(std*lam+K.epsilon())-0.5
        #clip+round
        tmp=K.round(K.clip(tmp,mmin,mmax))+0.5
        #second, scaler the modulus loss
        alpha=dot(w,tmp)/(dot(tmp,tmp)+K.epsilon())
        out=tmp*alpha
        return w+K.stop_gradient(out-w)
    return quantize