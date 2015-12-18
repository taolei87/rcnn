import random
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import HostFromGpu
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from theano.printing import debugprint

from utils import say
from .initialization import default_srng, default_rng, USE_XAVIER_INIT
from .initialization import set_default_rng_seed, random_init, create_shared
from .initialization import ReLU, sigmoid, tanh, softmax, linear, get_activation_by_name
from .advanced import StrCNN, AttentionLayer, BilinearAttentionLayer

class Dropout(object):
    def __init__(self, dropout_prob, srng=None, v2=False):
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2

    def forward(self, x):
        d = (1-self.dropout_prob) if not self.v2 else (1-self.dropout_prob)**0.5
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = x.shape
            )
        mask = T.cast(mask, theano.config.floatX) / d
        return x*mask

def apply_dropout(x, dropout_prob):
    return Dropout(dropout_prob).forward(x)

class Layer(object):

    def __init__(self, n_in, n_out, activation,
                            clip_gradients=False,
                            has_bias=True):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.has_bias = has_bias

        self.create_parameters()

        # not implemented yet
        if clip_gradients is True:
            raise Exception("gradient clip not implemented")

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.create_parameters_2(n_in, n_out, activation)

    def create_parameters_2(self, n_in, n_out, activation):
        if USE_XAVIER_INIT:
            if activation == ReLU:
                scale = np.sqrt(4.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            elif activation == softmax:
                scale = np.float64(0.001).astype(theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            else:
                scale = np.sqrt(2.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            W_vals = random_init((n_in,n_out), rng_type="normal") * scale
        else:
            W_vals = random_init((n_in,n_out))
            if activation == softmax:
                W_vals *= 0.001
            if activation == ReLU:
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.01
            else:
                b_vals = random_init((n_out,))
        self.W = create_shared(W_vals, name="W")
        if self.has_bias: self.b = create_shared(b_vals, name="b")

    def forward(self, x):
        if self.has_bias:
            return self.activation(
                    T.dot(x, self.W) + self.b
                )
        else:
            return self.activation(
                    T.dot(x, self.W)
                )

    @property
    def params(self):
        if self.has_bias:
            return [ self.W, self.b ]
        else:
            return [ self.W ]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())


class RecurrentLayer(Layer):

    def __init__(self, n_in, n_out, activation,
            clip_gradients=False):
        super(RecurrentLayer, self).__init__(
                n_in, n_out, activation,
                clip_gradients = clip_gradients
            )

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        # re-use the code in class Layer
        self.create_parameters_2(n_in + n_out, n_out, activation)

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        return activation(
                T.dot(x, self.W[:n_in]) + T.dot(h, self.W[n_in:]) + self.b
            )

    def forward_all(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        return h

class LSTM(Layer):

    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients

        self.in_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)


        self.internal_layers = [ self.input_layer, self.in_gate,
                                 self.forget_gate , self.out_gate ]

    def forward(self, x, hc):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[:, :n_out]
            h_tm1 = hc[:, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        in_t = self.in_gate.forward(x,h_tm1)
        forget_t = self.forget_gate.forward(x,h_tm1)
        out_t = self.out_gate.forward(x, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer.forward(x,h_tm1)
        h_t = out_t * T.tanh(c_t)

        if hc.ndim > 1:
            return T.concatenate([ c_t, h_t ], axis=1)
        else:
            return T.concatenate([ c_t, h_t ])

    def forward_all(self, x, h0=None, return_c=False):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*2,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out:]
        else:
            return h[:,self.n_out:]

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

class RCNN(Layer):

    def __init__(self, n_in, n_out, activation=tanh,
            order=2, has_outgate=False, mode=1, clip_gradients=False):

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


class CNN(Layer):

    def __init__(self, n_in, n_out, activation=tanh,
            order=1, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients

        internal_layers = self.internal_layers = [ ]
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, \
                    clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        self.bias = create_shared(random_init((n_out,)), name="bias")

    def forward(self, x, hc):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            if i == 0:
                c_i_t = layers[i].forward(x)
            else:
                c_i_t = c_im1_tm1 + layers[i].forward(x)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1

        h_t = activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

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



class EmbeddingLayer(object):

    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):

        if embs is not None:
            vocab_map = {}
            emb_vals = [ ]
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,))*0.001)

            emb_vals = np.vstack(emb_vals).astype(theano.config.floatX)
            self.vocab_map = vocab_map
        else:
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)

            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d))
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        self.embeddings = create_shared(emb_vals)
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_ids(self, words, filter_oov=False):
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )

    def forward(self, x):
        return self.embeddings[x]

    @property
    def params(self):
        return [ self.embeddings_trainable ]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())


def create_optimization_updates(
                cost, params, method="sgd",
                max_norm=5, updates=None, gradients=None,
                lr=0.01, eps=1e-8, rho=0.95,
                beta1=0.9, beta2=0.999):

    lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    eps = np.float64(eps).astype(theano.config.floatX)
    rho = theano.shared(np.float64(rho).astype(theano.config.floatX))
    beta1 = theano.shared(np.float64(beta1).astype(theano.config.floatX))
    beta2 = theano.shared(np.float64(beta2).astype(theano.config.floatX))

    gparams = T.grad(cost, params) if gradients is None else gradients

    g_norm = 0
    for g in gparams:
        g_norm = g_norm + g.norm(2)**2
    g_norm = T.sqrt(g_norm)

    # max_norm is useful for sgd
    if method != "sgd": max_norm = None

    if max_norm is not None and max_norm is not False:
        max_norm = theano.shared(np.float64(max_norm).astype(theano.config.floatX))
        shrink_factor = T.minimum(max_norm, g_norm + eps) / (g_norm + eps)
        gparams_clipped = [ ]
        for g in gparams:
            g = shrink_factor * g
            gparams_clipped.append(g)
        gparams = gparams_clipped

    if updates is None:
        updates = OrderedDict()

    gsums = create_accumulators(params) if method != "sgd" else None
    xsums = create_accumulators(params) if method != "sgd" and method != "adagrad" else None

    if method == "sgd":
        for p, g in zip(params, gparams):
            if is_subtensor_op(p):
                origin, _ = get_subtensor_op_inputs(p)
                updates[origin] = T.inc_subtensor(p, - lr*g)
            else:
                updates[p] = p - lr*g

    elif method == "adagrad":
        create_adagrad_updates(updates, params, gparams, gsums, lr, eps)

    elif method == "adadelta":
        create_adadelta_updates(updates, params, gparams, gsums, xsums, lr, eps, rho)

    elif method == "adam":
        create_adam_updates(updates, params, gparams, gsums, xsums, lr, eps, beta1, beta2)

    else:
        raise Exception("Unknown optim method: {}\n".format(method))

    if method == "adadelta":
        lr = rho

    return updates, lr, g_norm, gsums, xsums, max_norm

def is_subtensor_op(p):
    if hasattr(p, 'owner') and hasattr(p.owner, 'op'):
        return isinstance(p.owner.op, T.AdvancedSubtensor1) or \
               isinstance(p.owner.op, T.Subtensor)
    return False

def get_subtensor_op_inputs(p):
    origin, indexes = p.owner.inputs
    if hasattr(origin, 'owner') and hasattr(origin.owner, 'op') and \
            isinstance(origin.owner.op, HostFromGpu):
        origin = origin.owner.inputs[0]
        assert isinstance(origin, CudaNdarraySharedVariable)
    return origin, indexes

def get_similar_subtensor(matrix, indexes, param_op):
    '''
        So far there is only two possible subtensor operation used.
    '''
    if isinstance(param_op.owner.op, T.AdvancedSubtensor1):
        return matrix[indexes]
    else:
        # indexes is start index in this case
        return matrix[indexes:]


def create_accumulators(params):
    accums = [ ]
    for p in params:
        if is_subtensor_op(p):
            origin, _ = get_subtensor_op_inputs(p)
            acc = theano.shared(np.zeros_like(origin.get_value(borrow=True), \
                                 dtype=theano.config.floatX))
        else:
            acc = theano.shared(np.zeros_like(p.get_value(borrow=True), \
                                 dtype=theano.config.floatX))
        accums.append(acc)
    return accums

def create_adagrad_updates(updates, params, gparams, gsums, lr, eps):
    for p, g, acc in zip(params, gparams, gsums):
        if is_subtensor_op(p):
            origin, indexes = get_subtensor_op_inputs(p)
            #acc_slices = acc[indexes]
            acc_slices = get_similar_subtensor(acc, indexes, p)
            new_acc = acc_slices + g**2
            updates[acc] = T.set_subtensor(acc_slices, new_acc)
            updates[origin] = T.inc_subtensor(p, \
                    - lr * (g / T.sqrt(new_acc + eps)))
        else:
            new_acc = acc + g**2
            updates[acc] = new_acc
            updates[p] = p - lr * (g / T.sqrt(new_acc + eps))


def create_adadelta_updates(updates, params, gparams, gsums, xsums,\
                                lr, eps, rho):
    for p, g, gacc, xacc in zip(params, gparams, gsums, xsums):
        if is_subtensor_op(p):
            origin, indexes = get_subtensor_op_inputs(p)
            gacc_slices = gacc[indexes]
            xacc_slices = xacc[indexes]
            new_gacc = rho * gacc_slices + (1.0-rho) * g**2
            d = -T.sqrt((xacc_slices + eps)/(new_gacc + eps)) * g
            new_xacc = rho * xacc_slices + (1.0-rho) * d**2
            updates[gacc] = T.set_subtensor(gacc_slices, new_gacc)
            updates[xacc] = T.set_subtensor(xacc_slices, new_xacc)
            updates[origin] = T.inc_subtensor(p, d)
        else:
            new_gacc = rho * gacc + (1.0-rho) * g**2
            d = -T.sqrt((xacc + eps)/(new_gacc + eps)) * g
            new_xacc = rho * xacc + (1.0-rho) * d**2
            updates[gacc] = new_gacc
            updates[xacc] = new_xacc
            updates[p] = p + d

def create_adam_updates(updates, params, gparams, gsums, xsums, \
                            lr, eps, beta1, beta2):
    i = theano.shared(np.float64(0.0).astype(theano.config.floatX))
    i_t = i + 1.0
    omb1_t = 1.0 - beta1**i_t
    omb2_t = 1.0 - beta2**i_t
    lr_t = lr * (T.sqrt(omb2_t) / omb1_t)
    for p, g, m, v in zip(params, gparams, gsums, xsums):
        if is_subtensor_op(p):
            origin, indexes = get_subtensor_op_inputs(p)
            m_sub = m[indexes]
            v_sub = v[indexes]
            m_t = beta1*m_sub + (1.0-beta1)*g
            v_t = beta2*v_sub + (1.0-beta2)*T.sqr(g)
            g_t = m_t / (T.sqrt(v_t) + eps)
            updates[m] = T.set_subtensor(m_sub, m_t)
            updates[v] = T.set_subtensor(v_sub, v_t)
            updates[origin] = T.inc_subtensor(p, -lr_t*g_t)
        else:
            m_t = beta1*m + (1.0-beta1)*g
            v_t = beta2*v + (1.0-beta2)*T.sqr(g)
            g_t = m_t / (T.sqrt(v_t) + eps)
            updates[m] = m_t
            updates[v] = v_t
            updates[p] = p - lr_t*g_t
    updates[i] = i_t

