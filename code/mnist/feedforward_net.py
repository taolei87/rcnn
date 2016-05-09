import os
import sys
import gzip
import time
import math
import argparse
import cPickle as pickle

import numpy as np
import theano
import theano.tensor as T

from nn import Layer, softmax, ReLU, create_optimization_updates, apply_dropout
from utils import say

'''
    Load MNIST dataset. Code taken from Theano Deep Learning Tutorial:
        http://deeplearning.net/tutorial/
'''
def load_data(path):
    with gzip.open(path, 'rb') as f:
        try:
            train_set, dev_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, dev_set, test_set = pickle.load(f)
    # train_set, dev_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    train_x, train_y = train_set
    dev_x, dev_y = dev_set
    test_x, test_y = test_set

    train_x = theano.shared(np.asarray(train_x, dtype="float32"), borrow=True)
    dev_x = theano.shared(np.asarray(dev_x, dtype="float32"), borrow=True)
    test_x = theano.shared(np.asarray(test_x, dtype="float32"), borrow=True)

    train_y = theano.shared(np.asarray(train_y, dtype="float32"), borrow=True)
    dev_y = theano.shared(np.asarray(dev_y, dtype="float32"), borrow=True)
    test_y = theano.shared(np.asarray(test_y, dtype="float32"), borrow=True)

    train_y = T.cast(train_y, "int32")
    dev_y = T.cast(dev_y, "int32")
    test_y = T.cast(test_y, "int32")

    return (train_x, train_y), \
           (dev_x, dev_y), \
           (test_x, test_y)


class LogisticModel(object):
    def __init__(self, args, train_set, dev_set, test_set):
        self.args = args
        self.train_set, self.dev_set, self.test_set = train_set, dev_set, test_set

    '''
        Construct Theano computation graph
    '''
    def ready(self):
        args = self.args
        index = self.index = T.lscalar()
        x = self.x = T.fmatrix()
        y = self.y = T.ivector()

        dropout = self.dropout = theano.shared(np.float64(args.dropout).astype(
                        "float32"))

        n_d = args.hidden_dim
        layers = self.layers = [ ]
        for i in xrange(args.depth):
            l = Layer(
                        n_in = 28*28 if i == 0 else n_d,
                        n_out = n_d,
                        activation = ReLU
                    )
            layers.append(l)


        output_layer = self.output_layer = Layer(
                    n_in = n_d,
                    n_out = 10,
                    activation = softmax
                )

        h = x
        for l in layers:
            h = l.forward(h)
            h = apply_dropout(h, dropout)

        self.h_final = h

        # batch * 10
        probs = self.probs = output_layer.forward(h)

        # batch
        preds = self.preds = T.argmax(probs, axis=1)
        err = self.err = T.mean(T.cast(T.neq(preds, y), dtype="float32"))

        #
        loss = self.loss = -T.mean( T.log(probs[T.arange(y.shape[0]), y]) )
        #loss = self.loss = T.mean( T.nnet.categorical_crossentropy(
        #                            probs,
        #                            y
        #                    ))

        params = self.params = [ ]
        for l in layers + [ output_layer ]:
            for p in l.params:
                params.append(p)

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p**2)
            else:
                l2_cost += T.sum(p**2)
        l2_cost = l2_cost * args.l2_reg

        self.l2_cost = l2_cost
        self.cost = loss + l2_cost
        print "cost.dtype", self.cost.dtype

    def train(self):
        args = self.args
        train_x, train_y = self.train_set
        dev_x, dev_y = self.dev_set
        test_x, test_y = self.test_set

        updates, lr, gnorm, gsums, xsums, = create_optimization_updates(
                cost = self.cost,
                params = self.params,
                lr = args.learning_rate,
                rho = args.rho,
                beta1 = args.beta1,
                beta2 = args.beta2,
                momentum = args.momentum,
                gamma = args.gamma,
                method = args.learning
            )[:5]

        #gvals = [ updates[x] for x in updates if x in gsums ]
        #xvals = [ updates[x] for x in updates if x in xsums ]

        batch = args.batch
        index = self.index
        x = self.x
        y = self.y

        train_func = theano.function(
                inputs = [ index ],
                outputs = [ self.cost, gnorm ],
                givens = {
                    x: train_x[index*batch:(index+1)*batch],
                    y: train_y[index*batch:(index+1)*batch]
                },
                updates = updates
            )

        #debug_func = theano.function(
        #        inputs = [ index ],
        #        outputs = [ self.cost, gnorm ] + gvals + xvals,
        #        givens = {
        #            x: train_x[index*batch:(index+1)*batch],
        #            y: train_y[index*batch:(index+1)*batch]
        #        },
        #        updates = updates
        #    )

        dev_func = theano.function(
                inputs = [ index ],
                outputs = [ self.err, self.loss ],
                givens = {
                    x: dev_x[index*batch:(index+1)*batch],
                    y: dev_y[index*batch:(index+1)*batch]
                }
            )

        test_func = theano.function(
                inputs = [ index ],
                outputs = [ self.err, self.loss ],
                givens = {
                    x: test_x[index*batch:(index+1)*batch],
                    y: test_y[index*batch:(index+1)*batch]
                }
            )

        N = train_x.get_value(borrow=True).shape[0]
        num_batches = (N-1)/batch + 1
        processed = 0
        period = args.eval_period

        dropout_p = np.float64(args.dropout).astype("float32")
        decay_lr = args.decay_lr and args.learning.lower() != "adadelta" and \
                        args.learning.lower() != "adagrad"
        lr_0 = args.learning_rate
        iter_cnt = 0

        best_dev_err = 1.0

        self.print_norms()

        max_epochs = args.max_epochs
        for epoch in xrange(max_epochs):
            start_time = time.time()
            tot_cost = 0
            for i in xrange(num_batches):
                iter_cnt += 1
                if decay_lr:
                    lr.set_value(np.float32(lr_0/iter_cnt**0.5))

                cost, grad_norm = train_func(i)
                tot_cost += cost

                #debug
                if math.isnan(cost) or math.isnan(grad_norm):
                    say("NaN !!\n")
                    return

                ed = min(N, (i+1)*batch)
                prev = processed/period
                processed += ed-i*batch

                if (i == num_batches-1) or (processed/period > prev):
                    say("Epoch={:.1f} Sample={} cost={:.4f} |g|={:.2f}\t[{:.1f}m]\n".format(
                            epoch + (i+1.0)/num_batches,
                            processed,
                            tot_cost/(i+1),
                            float(grad_norm),
                            (time.time()-start_time)/60.0
                        ))
                    self.print_norms()
                    self.print_norms(xsums)
                    self.dropout.set_value(0.0)
                    dev_err, dev_loss = self.evaluate(dev_func, dev_x)
                    best_dev_err = min(best_dev_err, dev_err)
                    say("\tdev_err={:.4f} dev_loss={:.4f} best_dev={:.4f}\n".format(
                            dev_err, dev_loss, best_dev_err))
                    if dev_err == best_dev_err:
                        test_err, test_loss = self.evaluate(test_func, test_x)
                        say("\ttest_err={:.4f} test_loss={:.4f}\n".format(
                                test_err, test_loss))
                    say("\n")
                    self.dropout.set_value(dropout_p)

    def print_norms(self, lst=None):
        if lst is None: lst = self.params
        say("\t{}\n".format(
                [ "{:.4f}".format(np.linalg.norm(p.get_value(borrow=True))) \
                        for p in lst ]
            ))

    def evaluate(self, eval_func, data_x):
        args = self.args
        M = data_x.get_value(borrow=True).shape[0]
        num_batches = (M-1)/args.batch + 1

        tot_err, tot_loss = 0.0, 0.0
        for i in xrange(num_batches):
            err, loss = eval_func(i)
            tot_err += err
            tot_loss += loss

        return tot_err/num_batches, tot_loss/num_batches


def main(args):
    train_set, dev_set, test_set = load_data(args.data)
    model = LogisticModel(args, train_set, dev_set, test_set)
    model.ready()
    model.train()

if __name__=="__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--data",
            type = str,
            default = "mnist.pkl.gz",
            help = "path to data"
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adam",
            help = "learning method (sgd, adagrad, adam, ...)"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = "0.001",
            help = "learning rate"
        )
    argparser.add_argument("--depth",
            type = int,
            default = 2
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 500
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--eval_period",
            type = int,
            default = 10000,
            help = "evaluate on dev every period"
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.5
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 1e-5
        )
    argparser.add_argument("--rho",
            type = float,
            default = 0.95
        )
    argparser.add_argument("--beta1",
            type = float,
            default = 0.9
        )
    argparser.add_argument("--beta2",
            type = float,
            default = 0.999
        )
    argparser.add_argument("--momentum",
            type = float,
            default  = 0.0
        )
    argparser.add_argument("--gamma",
            type = float,
            default = 0.95
        )
    argparser.add_argument("--batch",
            type = int,
            default = 128,
            help = "mini-batch size"
        )
    argparser.add_argument("--decay_lr",
            type = int,
            default = 0
        )
    args = argparser.parse_args()
    main(args)
