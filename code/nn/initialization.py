'''
    This file implements various methods for initializing NN parameters

    @author: Tao Lei (taolei@csail.mit.edu)
'''

import random

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

'''
    whether to use Xavier initialization, as described in
        http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
'''
USE_XAVIER_INIT = False


'''
    Defaut random generators
'''
#random.seed(5817)
default_rng = np.random.RandomState(random.randint(0,9999))
#default_srng = T.shared_randomstreams.RandomStreams(default_rng.randint(9999))
default_mrng = MRG_RandomStreams(default_rng.randint(9999))
default_srng = default_mrng

'''
    Activation functions
'''
ReLU = lambda x: x * (x > 0)
sigmoid = T.nnet.sigmoid
tanh = T.tanh
softmax = T.nnet.softmax
linear = lambda x: x

def get_activation_by_name(name):
    if name.lower() == "relu":
        return ReLU
    elif name.lower() == "sigmoid":
        return sigmoid
    elif name.lower() == "tanh":
        return tanh
    elif name.lower() == "softmax":
        return softmax
    elif name.lower() == "none" or name.lower() == "linear":
        return linear
    else:
        raise Exception(
            "unknown activation type: {}".format(name)
          )

def set_default_rng_seed(seed):
    global default_rng, default_srng
    random.seed(seed)
    default_rng = np.random.RandomState(random.randint(0,9999))
    default_srng = T.shared_randomstreams.RandomStreams(default_rng.randint(9999))


'''
    Return initial parameter values of the specified size

    Inputs
    ------

        size            : size of the parameter, e.g. (100, 200) and (100,)
        rng             : random generator; the default is used if None
        rng_type        : the way to initialize the values
                            None    -- (default) uniform [-0.05, 0.05]
                            normal  -- Normal distribution with unit variance and zero mean
                            uniform -- uniform distribution with unit variance and zero mean
'''
def random_init(size, rng=None, rng_type=None):
    if rng is None: rng = default_rng
    if rng_type is None:
        #vals = rng.standard_normal(size)
        vals = rng.uniform(low=-0.05, high=0.05, size=size)

    elif rng_type == "normal":
        vals = rng.standard_normal(size)

    elif rng_type == "uniform":
        vals = rng.uniform(low=-3.0**0.5, high=3.0**0.5, size=size)

    else:
        raise Exception(
            "unknown random inittype: {}".format(rng_type)
          )

    return vals.astype(theano.config.floatX)


'''
    return a theano shared variable with initial values as vals
'''
def create_shared(vals, name=None):
    return theano.shared(vals, name=name)



