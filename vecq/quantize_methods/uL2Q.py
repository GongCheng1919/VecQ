import tensorflow as tf
class uL2Q:
    def __init__(self,bits=2):
        lambdas=[1.5950161625215948, 
                 0.99529008541347519, 
                 0.5872211503939504, 
                 0.3358904980253396, 
                 0.18944224088629152, 
                 0.10570877041455067, 
                 0.057082736023857356, 
                 0.030482181036739827]
        if bits>8:
            self.lam=6/(2**(bits)-1)
        else:
            self.lam=lambdas[bits-1]
        self.mmax=2**(bits-1)-1
        self.mmin=-2**(bits-1)
    # calculating means and standard derivation
    def quantize(self,d):
        self.beta=tf.reduce_mean(d)
        self.alpha=self.lam*tf.sqrt(tf.reduce_mean(tf.square(d - self.beta)))               
        # quantize
        self.fixed=(d-self.beta)/(self.alpha+1e-7)-0.5
        self.fixed=tf.clip_by_value(tf.round(self.fixed),self.mmin,self.mmax)+0.5
        # restore
        out=(self.fixed)*self.alpha+self.beta
        # ste
        return d+tf.stop_gradient(out-d)
    def __call__(self,w): 
        return self.quantize(w)