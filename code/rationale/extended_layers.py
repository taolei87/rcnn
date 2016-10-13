import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn import create_optimization_updates, get_activation_by_name, sigmoid, linear
from nn import EmbeddingLayer, Layer, RecurrentLayer, LSTM, RCNN, apply_dropout, default_rng
from nn import create_shared, random_init

class ExtRCNN(RCNN):

    def forward(self, x_t, mask_t, hc_tm1):
        hc_t = super(ExtRCNN, self).forward(x_t, hc_tm1)
        hc_t = mask_t * hc_t + (1-mask_t) * hc_tm1
        return hc_t

    def forward_all(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [ x, mask ],
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out*self.order:]
        else:
            return h[:,self.n_out*self.order:]

    def copy_params(self, from_obj):
        self.internal_layers = from_obj.internal_layers
        self.bias = from_obj.bias

class ExtLSTM(LSTM):

    def forward(self, x_t, mask_t, hc_tm1):
        hc_t = super(LSTM, self).forward(x_t, hc_tm1)
        hc_t = mask_t * hc_t + (1-mask_t) * hc_tm1
        return hc_t

    def forward_all(self, x, mask, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [ x, mask ],
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out*self.order:]
        else:
            return h[:,self.n_out*self.order:]

    def copy_params(self, from_obj):
        self.internal_layers = from_obj.internal_layers

class ZLayer(object):
    def __init__(self, n_in, n_hidden, activation):
        self.n_in, self.n_hidden, self.activation = \
                n_in, n_hidden, activation
        self.MRG_rng = MRG_RandomStreams()
        self.create_parameters()

    def create_parameters(self):
        n_in, n_hidden = self.n_in, self.n_hidden
        activation = self.activation

        self.w1 = create_shared(random_init((n_in,)), name="w1")
        self.w2 = create_shared(random_init((n_hidden,)), name="w2")
        bias_val = random_init((1,))[0]
        self.bias = theano.shared(np.cast[theano.config.floatX](bias_val))
        rlayer = RCNN((n_in+1), n_hidden, activation=activation, order=2)
        self.rlayer = rlayer
        self.layers = [ rlayer ]

    def forward(self, x_t, z_t, h_tm1, pz_tm1):

        print "z_t", z_t.ndim

        pz_t = sigmoid(
                    T.dot(x_t, self.w1) +
                    T.dot(h_tm1[:,-self.n_hidden:], self.w2) +
                    self.bias
                )

        xz_t =  T.concatenate([x_t, z_t.reshape((-1,1))], axis=1)
        h_t = self.rlayer.forward(xz_t, h_tm1)

        # batch
        return h_t, pz_t

    def forward_all(self, x, z):
        assert x.ndim == 3
        assert z.ndim == 2
        xz = T.concatenate([x, z.dimshuffle((0,1,"x"))], axis=2)
        h0 = T.zeros((1, x.shape[1], self.n_hidden), dtype=theano.config.floatX)
        h = self.rlayer.forward_all(xz)
        h_prev = T.concatenate([h0, h[:-1]], axis=0)
        assert h.ndim == 3
        assert h_prev.ndim == 3
        pz = sigmoid(
                T.dot(x, self.w1) +
                T.dot(h_prev, self.w2) +
                self.bias
            )
        assert pz.ndim == 2
        return pz

    def sample(self, x_t, z_tm1, h_tm1):

        print "z_tm1", z_tm1.ndim, type(z_tm1)

        pz_t = sigmoid(
                    T.dot(x_t, self.w1) +
                    T.dot(h_tm1[:,-self.n_hidden:], self.w2) +
                    self.bias
                )

        # batch
        pz_t = pz_t.ravel()
        z_t = T.cast(self.MRG_rng.binomial(size=pz_t.shape,
                                        p=pz_t), theano.config.floatX)

        xz_t = T.concatenate([x_t, z_t.reshape((-1,1))], axis=1)
        h_t = self.rlayer.forward(xz_t, h_tm1)

        return z_t, h_t

    def sample_all(self, x):
        h0 = T.zeros((x.shape[1], self.n_hidden*(self.rlayer.order+1)), dtype=theano.config.floatX)
        z0 = T.zeros((x.shape[1],), dtype=theano.config.floatX)
        ([ z, h ], updates) = theano.scan(
                            fn = self.sample,
                            sequences = [ x ],
                            outputs_info = [ z0, h0 ]
                    )
        assert z.ndim == 2
        return z, updates

    @property
    def params(self):
        return [ x for layer in self.layers for x in layer.params ] + \
               [ self.w1, self.w2, self.bias ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.w1.set_value(param_list[-3].get_value())
        self.w2.set_value(param_list[-2].get_value())
        self.bias.set_value(param_list[-1].get_value())

