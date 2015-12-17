
import numpy as np
import theano
import theano.tensor as T

from .initialization import random_init, create_shared
from .initialization import ReLU


class StrCNN:

    def __init__(self, n_in, n_out, activation=None, decay=0.0, order=2, use_all_grams=True):
        self.n_in = n_in
        self.n_out = n_out
        self.order = order
        self.use_all_grams = use_all_grams
        self.decay = theano.shared(np.float64(decay).astype(theano.config.floatX))
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation

        self.create_parameters()

    def create_parameters(self):
        n_in, n_out = self.n_in, self.n_out
        #rng_type = "uniform"
        #scale = 1.0/self.n_out**0.5
        rng_type = None
        scale = 1.0
        self.P = create_shared(random_init((n_in, n_out), rng_type=rng_type)*scale, name="P")
        self.Q = create_shared(random_init((n_in, n_out), rng_type=rng_type)*scale, name="Q")
        self.R = create_shared(random_init((n_in, n_out), rng_type=rng_type)*scale, name="R")
        self.O = create_shared(random_init((n_out, n_out), rng_type=rng_type)*scale, name="O")
        if self.activation == ReLU:
            self.b = create_shared(np.ones(n_out, dtype=theano.config.floatX)*0.01, name="b")
        else:
            self.b = create_shared(random_init((n_out,)), name="b")

    def forward(self, x_t, f1_tm1, s1_tm1, f2_tm1, s2_tm1, f3_tm1):
        P, Q, R, decay = self.P, self.Q, self.R, self.decay
        f1_t = T.dot(x_t, P)
        s1_t = s1_tm1 * decay + f1_t
        f2_t = T.dot(x_t, Q) * s1_tm1
        s2_t = s2_tm1 * decay + f2_t
        f3_t = T.dot(x_t, R) * s2_tm1
        return f1_t, s1_t, f2_t, s2_t, f3_t

    def forward_all(self, x, v0=None):
        if v0 is None:
            if x.ndim > 1:
                v0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                v0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        ([f1, s1, f2, s2, f3], updates) = theano.scan(
                        fn = self.forward,
                        sequences = x,
                        outputs_info = [ v0, v0, v0, v0, v0 ]
                )
        if self.order == 3:
            h = f1+f2+f3 if self.use_all_grams else f3
        elif self.order == 2:
            h = f1+f2 if self.use_all_grams else f2
        elif self.order == 1:
            h = f1
        else:
            raise ValueError(
                    "Unsupported order: {}".format(self.order)
                )
        return self.activation(T.dot(h, self.O) + self.b)

    @property
    def params(self):
        if self.order == 3:
            return [ self.b, self.O, self.P, self.Q, self.R ]
        elif self.order == 2:
            return [ self.b, self.O, self.P, self.Q ]
        elif self.order == 1:
            return [ self.b, self.O, self.P ]
        else:
            raise ValueError(
                    "Unsupported order: {}".format(self.order)
                )

    @params.setter
    def params(self, param_list):
        for p, q in zip(self.params, param_list):
            p.set_value(q.get_value())



class AttentionLayer:
    def __init__(self, n_d, activation):
        self.n_d = n_d
        self.activation = activation
        self.create_parameters()

    def create_parameters(self):
        n_d = self.n_d
        self.W1_c = create_shared(random_init((n_d, n_d)), name="W1_c")
        self.W1_h = create_shared(random_init((n_d, n_d)), name="W1_h")
        self.w = create_shared(random_init((n_d,)), name="w")
        self.W2_r = create_shared(random_init((n_d, n_d)), name="W1_r")
        self.W2_h = create_shared(random_init((n_d, n_d)), name="W1_h")
        self.lst_params = [ self.W1_h, self.W1_c, self.W2_h, self.W2_r, self.w ]

    def forward(self, h_before, h_after_tm1, C, mask=None):
        # C is batch*len*d
        # h is batch*d

        M = self.activation(
                T.dot(C, self.W1_c) + T.dot(h_before, self.W1_h).dimshuffle((0,'x',1))
            )

        # batch*len*1
        alpha = T.nnet.softmax(
                    T.dot(M, self.w)
                )
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        if mask is not None:
            eps = 1e-8
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            alpha = alpha*mask.dimshuffle((0,1,'x'))
            alpha = alpha / (T.sum(alpha, axis=1).dimshuffle((0,1,'x'))+eps)

        # batch * d
        r = T.sum(C*alpha, axis=1)

        # batch * d
        h_after = self.activation(
                T.dot(r, self.W2_r) + T.dot(h_before, self.W2_h)
            )
        return h_after

    def one_step(self, h_before, h_after_tm1, r):
        h_after = self.activation(
                T.dot(r, self.W2_r) + T.dot(h_before, self.W2_h)
            )
        return h_after

    def forward_all(self, x, C, mask=None):
        # batch*len2*d
        C2 = T.dot(C, self.W1_c).dimshuffle(('x',0,1,2))
        # len1*batch*d
        x2 = T.dot(x, self.W1_h).dimshuffle((0,1,'x',2))
        # len1*batch*len2*d
        M = self.activation(C2 + x2)

        # len1*batch*len2*1
        alpha = T.nnet.softmax(
                    T.dot(M, self.w).reshape((-1, C.shape[1]))
                )
        alpha = alpha.reshape((x.shape[0],x.shape[1],C.shape[1],1))
        if mask is not None:
            # mask is batch*len2
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            mask = mask.dimshuffle(('x',0,1,'x'))
            alpha = alpha*mask
            alpha = alpha / (T.sum(alpha, axis=2).dimshuffle((0,1,2,'x')) + 1e-8)

        # len1*batch*d
        r = T.sum(C.dimshuffle(('x',0,1,2)) * alpha, axis=2)

        # len1*batch*d
        h = self.activation(
                T.dot(r, self.W2_r) + T.dot(x, self.W2_h)
            )

        '''
            Use scan when recurrent attention is implemented
        '''
        #func = lambda h, r: self.one_step(h, None, r)
        #h, _ = theano.scan(
        #            fn = func,
        #            sequences = [ x, r ]
        #        )
        return h

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())



class BilinearAttentionLayer:
    def __init__(self, n_d, activation, weighted_output=True):
        self.n_d = n_d
        self.activation = activation
        self.weighted_output = weighted_output
        self.create_parameters()

    def create_parameters(self):
        n_d = self.n_d
        self.P = create_shared(random_init((n_d, n_d)), name="P")
        self.W_r = create_shared(random_init((n_d, n_d)), name="W_r")
        self.W_h = create_shared(random_init((n_d, n_d)), name="W_h")
        self.b = create_shared(random_init((n_d,)), name="b")
        self.lst_params = [ self.P, self.W_r, self.W_h, self.b ]

    def forward(self, h_before, h_after_tm1, C, mask=None):
        # C is batch*len*d
        # h is batch*d
        # mask is batch*len

        # batch*1*d
        #M = T.dot(h_before, self.P).dimshuffle((0,'x',1))
        M = T.dot(h_before, self.P).reshape((h_before.shape[0], 1, h_before.shape[1]))

        # batch*len*1
        alpha = T.nnet.softmax(
                    T.sum(C * M, axis=2)
                )
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        if mask is not None:
            eps = 1e-8
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            alpha = alpha*mask.dimshuffle((0,1,'x'))
            alpha = alpha / (T.sum(alpha, axis=1).dimshuffle((0,1,'x'))+eps)

        # batch * d
        r = T.sum(C*alpha, axis=1)

        # batch * d
        if self.weighted_output:
            beta = T.nnet.sigmoid(
                    T.dot(r, self.W_r) + T.dot(h_before, self.W_h) + self.b
                )
            h_after = beta*h_before + (1.0-beta)*r
        else:
            h_after = self.activation(
                    T.dot(r, self.W_r) + T.dot(h_before, self.W_h) + self.b
                )
        return h_after

    def forward_all(self, x, C, mask=None):
        # x is len1*batch*d
        # C is batch*len2*d
        # mask is batch*len2

        C2 = C.dimshuffle(('x',0,1,2))

        # batch*len1*d
        M = T.dot(x, self.P).dimshuffle((1,0,2))
        # batch*d*len2
        C3 = C.dimshuffle((0,2,1))

        alpha = T.batched_dot(M, C3).dimshuffle((1,0,2))
        alpha = T.nnet.softmax(
                    alpha.reshape((-1, C.shape[1]))
                )
        alpha = alpha.reshape((x.shape[0],x.shape[1],C.shape[1],1))
        if mask is not None:
            # mask is batch*len1
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            mask = mask.dimshuffle(('x',0,1,'x'))
            alpha = alpha*mask
            alpha = alpha / (T.sum(alpha, axis=2).dimshuffle((0,1,2,'x')) + 1e-8)

        # len1*batch*d
        r = T.sum(C2*alpha, axis=2)

        # len1*batch*d
        if self.weighted_output:
            beta = T.nnet.sigmoid(
                        T.dot(r, self.W_r) + T.dot(x, self.W_h) + self.b
                    )
            h = beta*x + (1.0-beta)*r
        else:
            h = self.activation(
                    T.dot(r, self.W_r) + T.dot(x, self.W_h) + self.b
                )

        '''
            Use scan when recurrent attention is implemented
        '''
        #func = lambda h, r: self.one_step(h, None, r)
        #h, _ = theano.scan(
        #            fn = func,
        #            sequences = [ x, r ]
        #        )
        return h


    def forward_all2(self, x, C, mask=None):
        # x is len1*batch*d
        # C is batch*len2*d
        # mask is batch*len2

        # len1*batch*1*d
        M = T.dot(x, self.P).dimshuffle((0,1,'x',2))
        # 1*batch*len2*d
        C2 = C.dimshuffle(('x',0,1,2))

        # len1*batch*len2*1
        alpha = T.nnet.softmax(
                    T.sum(C2*M, axis=3).reshape((-1, C.shape[1]))
                )
        alpha = alpha.reshape((x.shape[0],x.shape[1],C.shape[1],1))
        if mask is not None:
            # mask is batch*len1
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            mask = mask.dimshuffle(('x',0,1,'x'))
            alpha = alpha*mask
            alpha = alpha / (T.sum(alpha, axis=2).dimshuffle((0,1,2,'x')) + 1e-8)

        # len1*batch*d
        r = T.sum(C2*alpha, axis=2)

        # len1*batch*d
        if self.weighted_output:
            beta = T.nnet.sigmoid(
                        T.dot(r, self.W_r) + T.dot(x, self.W_h) + self.b
                    )
            h = beta*x + (1.0-beta)*r
        else:
            h = self.activation(
                    T.dot(r, self.W_r) + T.dot(x, self.W_h) + self.b
                )

        '''
            Use scan when recurrent attention is implemented
        '''
        #func = lambda h, r: self.one_step(h, None, r)
        #h, _ = theano.scan(
        #            fn = func,
        #            sequences = [ x, r ]
        #        )
        return h


    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())



