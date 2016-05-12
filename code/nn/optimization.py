'''
    This file implements various optimization methods, including
        -- SGD with gradient norm clipping
        -- AdaGrad
        -- AdaDelta
        -- Adam

    Transparent to switch between CPU / GPU.

    @author: Tao Lei (taolei@csail.mit.edu)
'''

import random
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import HostFromGpu
from theano.sandbox.cuda.var import CudaNdarraySharedVariable
from theano.printing import debugprint

from .initialization import default_mrng

def create_optimization_updates(
                cost, params, method="sgd",
                max_norm=5, updates=None, gradients=None,
                lr=0.01, eps=None, rho=0.99, gamma=0.999,
                beta1=0.9, beta2=0.999, momentum=0.0):

    _momentum = momentum
    lr = theano.shared(np.float64(lr).astype(theano.config.floatX))
    rho = theano.shared(np.float64(rho).astype(theano.config.floatX))
    beta1 = theano.shared(np.float64(beta1).astype(theano.config.floatX))
    beta2 = theano.shared(np.float64(beta2).astype(theano.config.floatX))
    momentum = theano.shared(np.float64(momentum).astype(theano.config.floatX))
    gamma = theano.shared(np.float64(gamma).astype(theano.config.floatX))

    if eps is None:
        eps = 1e-8 if method.lower() != "esgd" else 1e-4
    eps = np.float64(eps).astype(theano.config.floatX)

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

    gsums = create_accumulators(params) if method != "sgd" or _momentum > 0.0 else \
                [ None for p in params ]
    xsums = create_accumulators(params) if method != "sgd" and method != "adagrad" else None

    if method == "sgd":
        create_sgd_updates(updates, params, gparams, gsums, lr, momentum)

    elif method == "adagrad":
        create_adagrad_updates(updates, params, gparams, gsums, lr, eps)

    elif method == "adadelta":
        create_adadelta_updates(updates, params, gparams, gsums, xsums, lr, eps, rho)

    elif method == "adam":
        create_adam_updates(updates, params, gparams, gsums, xsums, lr, eps, beta1, beta2)

    elif method == "esgd":
        create_esgd_updates(updates, params, gparams, gsums, xsums, lr, eps, gamma, momentum)

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

def create_sgd_updates(updates, params, gparams, gsums, lr, momentum):
    has_momentum = momentum.get_value() > 0.0
    for p, g, acc in zip(params, gparams, gsums):
        if is_subtensor_op(p):
            origin, indexes = get_subtensor_op_inputs(p)
            if has_momentum:
                acc_slices = get_similar_subtensor(acc, indexes, p)
                new_acc = acc_slices*momentum + g
                updates[acc] = T.set_subtensor(acc_slices, new_acc)
            else:
                new_acc = g
            updates[origin] = T.inc_subtensor(p, - lr * new_acc)
        else:
            if has_momentum:
                new_acc = acc*momentum + g
                updates[acc] = new_acc
            else:
                new_acc = g
            updates[p] = p - lr * new_acc

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
            #updates[p] = p - lr * (g / (T.sqrt(new_acc) + eps))
            # which one to use?

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

def create_esgd_updates(updates, params, gparams, gsums, xsums, lr, eps, gamma, momentum):
    has_momentum = momentum.get_value() > 0.0
    samples = [ default_mrng.normal(size=p.shape, avg=0, std=1,
                    dtype=theano.config.floatX) for p in params ]
    HVs = T.Lop(gparams, params, samples)

    i = theano.shared(np.float64(0.0).astype(theano.config.floatX))
    i_t = i + 1.0
    omg_t = 1.0 - gamma**i_t
    for p, g, m, D, Hv in zip(params, gparams, gsums, xsums, HVs):
        if is_subtensor_op(p):
            raise Exception("ESGD subtensor update not implemented!")
        else:
            D_t = D * gamma + T.sqr(Hv) * (1.0-gamma)
            if has_momentum:
                m_t = m*momentum + g
                updates[m] = m_t
            else:
                m_t = g
            g_t = m_t / ( T.sqrt(D_t/omg_t + eps) )
            #g_t = m_t / ( T.sqrt(D_t + eps) )
            updates[D] = D_t
            updates[p] = p - lr*g_t
    updates[i] = i_t
