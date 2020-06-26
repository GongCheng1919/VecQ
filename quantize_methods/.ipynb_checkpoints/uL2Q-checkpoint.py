import tensorflow as tf
def uL2Q(bits=2):
    lambdas=[1.5950161625215948, 
             0.99529008541347519, 
             0.5872211503939504, 
             0.3358904980253396, 
             0.18944224088629152, 
             0.10570877041455067, 
             0.057082736023857356, 
             0.030482181036739827]
    if bits>8:
        lam=6/(2**(bits)-1)
    else:
        lam=lambdas[bits-1]
    mmax=2**(bits-1)-1
    mmin=-2**(bits-1)
    # calculating means and standard derivation
    def func(d):
        mean=tf.reduce_mean(d)
        std=lam*tf.sqrt(tf.reduce_mean(tf.square(d - mean)))               
        # quantize
        tmp=(d-mean)/(std+1e-7)-0.5
        tmp=tf.clip_by_value(tf.round(tmp),mmin,mmax)
        # restore
        out=(tmp+0.5)*std+mean
        # ste
        return d+tf.stop_gradient(out-d)
    return func