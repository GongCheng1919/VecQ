import keras.backend as K
class VecQ:
    def __init__(self,bits=2):
        lambdas=[1.0000, 0.9957, 0.5860, 0.3352, 0.1881, 0.1041, 0.0569, 0.0308]
        self.bits=bits
        if self.bits>8:
            self.lam=6/(2**(self.bits)-1)
        else:
            self.lam=lambdas[self.bits-1]
        self.mmax=2**(self.bits-1)-1
        self.mmin=-2**(self.bits-1)
    # calculating means and standard derivation
    def dot(self,a,b):
        return K.sum(a*b)
    def quantize(self,w):
        std=K.std(w)
        self.fixed=w/(std*self.lam+K.epsilon())-0.5
        #clip+round, orientation loss
        self.fixed=K.round(K.clip(self.fixed,self.mmin,self.mmax))+0.5
        #second, modulus loss
        self.alpha=self.dot(w,self.fixed)/(self.dot(self.fixed,self.fixed)+K.epsilon())
        out=self.fixed*self.alpha
        return w+K.stop_gradient(out-w)
    def __call__(self,w):
        return self.quantize(w)