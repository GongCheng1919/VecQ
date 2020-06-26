from keras.layers import Conv2D,Dense,DepthwiseConv2D
class Conv2D_Q(Conv2D):
    def __init__(self,kq=None,bq=None,aq=None,after_activation=None,**kwargs):
        self.kq=kq
        self.bq=bq
        self.aq=aq
        self.after_activation=after_activation
        super(Conv2D_Q, self).__init__(**kwargs)
        if self.activation is not None and self.after_activation is not None:
            raise ValueError("activation and after_activation is conflict.")
    def call(self,w):
        if self.kq is not None:
            self.kernel=self.kq(self.kernel)
        if self.use_bias and self.bq is not None:
            self.bias=self.bq(self.bias)
        output=super(Conv2D_Q, self).call(w)
        if self.aq is not None:
            output=self.aq(output)
        if self.after_activation is not None:
            output=self.after_activation(out)
        return output
class Dense_Q(Dense):
    def __init__(self,kq=None,bq=None,aq=None,after_activation=None,**kwargs):
        self.kq=kq
        self.bq=bq
        self.aq=aq
        self.after_activation=after_activation
        super(Dense_Q, self).__init__(**kwargs)
        if self.activation is not None and self.after_activation is not None:
            raise ValueError("activation and after_activation is conflict.")
    def call(self,w):
        if self.kq is not None:
            self.kernel=self.kq(self.kernel)
        if self.use_bias and self.bq is not None:
            self.bias=self.bq(self.bias)
        output=super(Dense_Q, self).call(w)
        if self.aq is not None:
            output=self.aq(output)
        if self.after_activation is not None:
            output=self.after_activation(out)
        return output
class DepthwiseConv2D_Q(DepthwiseConv2D):
    def __init__(self,kq=None,bq=None,aq=None,after_activation=None,**kwargs):
        self.kq=kq
        self.bq=bq
        self.aq=aq
        self.after_activation=after_activation
        super(DepthwiseConv2D_Q, self).__init__(**kwargs)
        if self.activation is not None and self.after_activation is not None:
            raise ValueError("activation and after_activation is conflict.")
    def call(self,w, training=None):
        if self.kq is not None:
            self.depthwise_kernel=self.kq(self.depthwise_kernel)
        if self.use_bias and self.bq is not None:
            self.bias=self.bq(self.bias)
        output=super(DepthwiseConv2D_Q, self).call(w, training=training)
        if self.aq is not None:
            output=self.aq(output)
        if self.after_activation is not None:
            output=self.after_activation(out)
        return output