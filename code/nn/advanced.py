'''
    This file contains implementations of advanced NN components, including
      -- Attention layer (two versions)
      -- StrCNN: non-consecutive & non-linear CNN
      -- RCNN: recurrent convolutional network

    Sequential layers (recurrent/convolutional) has two forward methods implemented:
        -- forward(x_t, h_tm1):  one step of forward given input x and previous
                                 hidden state h_tm1; return next hidden state

        -- forward_all(x, h_0):  apply successively steps given all inputs and
                                 initial hidden state, and return all hidden
                                 states h1, ..., h_n

    @author: Tao Lei (taolei@csail.mit.edu)
'''

import numpy as np
import theano
import theano.tensor as T

from .initialization import random_init, create_shared
from .initialization import ReLU, tanh, linear, sigmoid
from .basic import Layer, RecurrentLayer

'''
    This class implements the non-consecutive, non-linear CNN model described in
        Molding CNNs for text (http://arxiv.org/abs/1508.04112)
'''
class StrCNN(Layer):

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
        rng_type = "uniform"
        scale = 1.0/self.n_out**0.5
        #rng_type = None
        #scale = 1.0
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


'''
    This class implements the *basic* attention layer described in
        Reasoning about Entailment with Neural Attention (http://arxiv.org/abs/1509.06664)

    This layer is uni-directional and non-recurrent.
'''
class AttentionLayer(Layer):
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

    '''
        One step of attention activation.

        Inputs
        ------

        h_before        : the state before attention at time/position t
        h_after_tm1     : the state after attention at time/position t-1; not used
                          because the current attention implementation is not
                          recurrent
        C               : the context to pay attention to
        mask            : which positions are valid for attention; specify this when
                          some tokens in the context are non-meaningful tokens such
                          as paddings

        Outputs
        -------

        return the state after attention at time/position t
    '''
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

    '''
        Can change this when recurrent attention is needed.
    '''
    def one_step(self, h_before, h_after_tm1, r):
        h_after = self.activation(
                T.dot(r, self.W2_r) + T.dot(h_before, self.W2_h)
            )
        return h_after

    '''
        Apply the attention-based activation to all input tokens x_1, ..., x_n

        Return the post-activation representations
    '''
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
            The current version is non-recurrent, so theano scan is not needed.
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


'''
    This class implements the attention layer described in
        A Neural Attention Model for Abstractive Sentence Summarization
        (http://arxiv.org/pdf/1509.00685.pdf)

    This layer is uni-directional and non-recurrent.
'''
class BilinearAttentionLayer(Layer):
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

    '''
        One step of attention activation.

        Inputs
        ------

        h_before        : the state before attention at time/position t
        h_after_tm1     : the state after attention at time/position t-1; not used
                          because the current attention implementation is not
                          recurrent
        C               : the context to pay attention to
        mask            : which positions are valid for attention; specify this when
                          some tokens in the context are non-meaningful tokens such
                          as paddings

        Outputs
        -------

        return the state after attention at time/position t
    '''
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

    '''
        Apply the attention-based activation to all input tokens x_1, ..., x_n

        Return the post-activation representations
    '''
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
            The current version is non-recurrent, so theano scan is not needed.
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

'''
    This class implements the recurrent convolutional network model described in
        Retrieving Similar Questions with Recurrent Convolutional Models
        (http://arxiv.org/abs/1512.05726)
'''
class RCNN(Layer):

    '''
        RCNN

        Inputs
        ------

            order           : CNN feature width
            has_outgate     : whether to add a output gate as in LSTM; this can be
                              useful for language modeling
            mode            : 0 if non-linear filter; 1 if linear filter (default)
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            order=1, has_outgate=False, mode=1, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients
        self.has_outgate = has_outgate
        self.mode = mode

        internal_layers = self.internal_layers = [ ]
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, \
                    clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        internal_layers.append(forget_gate)

        self.bias = create_shared(random_init((n_out,)), name="bias")

        if has_outgate:
            self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
            self.internal_layers += [ self.out_gate ]

    '''
        One step of recurrent

        Inputs
        ------

            x           : input token at current time/position t
            hc          : hidden/visible states at time/position t-1

        Outputs
        -------

            return hidden/visible states at time/position t
    '''
    def forward(self, x, hc):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t + c_im1_tm1)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    '''
        Apply recurrent steps to input of all positions/time

        Inputs
        ------

            x           : input tokens x_1, ... , x_n
            h0          : initial states
            return_c    : whether to return hidden states in addition to visible
                          state

        Outputs
        -------

            return visible states (and hidden states) of all positions/time
    '''
    def forward_all(self, x, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out*self.order:]
        else:
            return h[:,self.n_out*self.order:]

    def forward2(self, x, hc, f_tm1):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t + c_im1_tm1)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1), forget_t
        else:
            return T.concatenate(lst), forget_t

    def get_input_gate(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
                f0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
                f0 = T.zeros((self.n_out,), dtype=theano.config.floatX)

        [h, f], _ = theano.scan(
                    fn = self.forward2,
                    sequences = x,
                    outputs_info = [ h0,f0 ]
                )
        return 1.0-f

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ] + [ self.bias ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.bias.set_value(param_list[-1].get_value())


