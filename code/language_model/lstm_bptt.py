
import sys
import os
import argparse
import time
import random
import math

import numpy as np
import theano
import theano.tensor as T

import nn
from nn import Dropout, EmbeddingLayer, RecurrentLayer, Layer, LSTM, apply_dropout
from nn import get_activation_by_name, create_optimization_updates
from nn.evaluation import evaluate_average

from utils import say

def read_corpus(path, max_sent=-1, eos="</s>"):
    data = [ ]
    with open(path) as fin:
        sent_cnt = 0
        for line in fin:
            sent_cnt += 1
            data += line.split() + [ eos ]
            if max_sent > 0 and sent_cnt == max_sent: break
    return data

def create_batches(data_text, map_to_ids, batch_size):
    data_ids = map_to_ids(data_text)
    N = len(data_ids)
    L = ((N-1)/batch_size) * batch_size
    x = np.copy(data_ids[:L].reshape(batch_size,-1).T)
    y = np.copy(data_ids[1:L+1].reshape(batch_size,-1).T)
    return x, y

class SimpleRNN(object):
    def __init__(self):
        return

    def ready(self, args, train):
        # len * batch
        self.idxs = T.imatrix()
        self.idys = T.imatrix()
        self.init_state = T.matrix(dtype=theano.config.floatX)

        dropout_prob = np.float64(args["dropout"]).astype(theano.config.floatX)
        self.dropout = theano.shared(dropout_prob)

        self.n_d = args["hidden_dim"]

        embedding_layer = EmbeddingLayer(
                n_d = self.n_d,
                vocab = set(w for w in train)
            )
        self.n_V = embedding_layer.n_V

        say("Vocab size: {}\tHidden dim: {}\n".format(
                self.n_V, self.n_d
            ))

        activation = get_activation_by_name(args["activation"])

        rnn_layer = LSTM(
                 n_in = self.n_d,
                 n_out = self.n_d,
                 activation = activation
            )

        output_layer = Layer(
                n_in = self.n_d,
                n_out = self.n_V,
                activation = T.nnet.softmax,
            )

        # (len*batch) * n_d
        x_flat = embedding_layer.forward(self.idxs.ravel())

        # len * batch * n_d
        x = apply_dropout(x_flat, self.dropout)
        x = x.reshape( (self.idxs.shape[0], self.idxs.shape[1], self.n_d) )

        # len * batch * (n_d+n_d)
        h = rnn_layer.forward_all(x, self.init_state, return_c=True)

        self.last_state = h[-1]
        h = h[:,:,self.n_d:]
        h = apply_dropout(h, self.dropout)

        self.p_y_given_x = output_layer.forward(h.reshape(x_flat.shape))

        idys = self.idys.ravel()
        self.nll = -T.log(self.p_y_given_x[T.arange(idys.shape[0]), idys])
        #self.nll = T.nnet.categorical_crossentropy(
        #                self.p_y_given_x,
        #                idys
        #            )

        self.layers = [ embedding_layer, rnn_layer, output_layer ]
        #self.params = [ x_flat ] + rnn_layer.params + output_layer.params
        self.params = embedding_layer.params + rnn_layer.params + output_layer.params
        self.num_params = sum(len(x.get_value(borrow=True).ravel())
                                for l in self.layers for x in l.params)
        say("# of params in total: {}\n".format(self.num_params))

    def train(self, args, train, dev, test=None):
        embedding_layer = self.layers[0]

        dropout_prob = np.float64(args["dropout"]).astype(theano.config.floatX)
        batch_size = args["batch_size"]
        unroll_size = args["unroll_size"]

        train = create_batches(train, embedding_layer.map_to_ids, batch_size)

        dev = create_batches(dev, embedding_layer.map_to_ids, batch_size)

        if test is not None:
            test = create_batches(test, embedding_layer.map_to_ids, batch_size)

        cost = T.sum(self.nll) / self.idxs.shape[1]
        updates, lr, gnorm = create_optimization_updates(
                cost = cost,
                params = self.params,
                lr = args["learning_rate"],
                beta1 = args["beta1"],
                beta2 = args["beta2"],
                rho = args["rho"],
                momentum = args["momentum"],
                gamma = args["gamma"],
                eps = args["eps"],
                method = args["learning"]
            )[:3]
        #if args["learning"] == "adadelta":
        #    lr.set_value(args["learning_rate"])

        train_func = theano.function(
                inputs = [ self.idxs, self.idys, self.init_state ],
                outputs = [cost, self.last_state, gnorm ],
                updates = updates
            )
        eval_func = theano.function(
                inputs = [ self.idxs, self.idys, self.init_state ],
                outputs = [self.nll, self.last_state ]
            )

        N = (len(train[0])-1)/unroll_size + 1
        say(" train: {} tokens, {} mini-batches\n".format(
                len(train[0].ravel()), N
            ))
        say(" dev: {} tokens\n".format(len(dev[0].ravel())))

        say("\tp_norm: {}\n".format(
                self.get_pnorm_stat()
            ))

        decay_lr = args["decay_lr"] and args["learning"].lower() != "adadelta" and \
                    args["learning"].lower() != "adagrad"
        lr_0 = args["learning_rate"]
        iter_cnt = 0

        unchanged = 0
        best_dev = 1e+10
        start_time = 0
        max_epoch = args["max_epoch"]
        for epoch in xrange(max_epoch):
            if unchanged > 5: break
            start_time = time.time()

            prev_state = np.zeros((batch_size, self.n_d*2),
                            dtype=theano.config.floatX)

            train_loss = 0.0
            for i in xrange(N):
                # get current batch
                x = train[0][i*unroll_size:(i+1)*unroll_size]
                y = train[1][i*unroll_size:(i+1)*unroll_size]

                iter_cnt += 1
                if decay_lr:
                    lr.set_value(np.float32(lr_0/iter_cnt**0.5))
                cur_loss, prev_state, grad_norm = train_func(x, y, prev_state)
                train_loss += cur_loss/len(x)

                if math.isnan(cur_loss) or math.isnan(grad_norm):
                    say("\nNaN !!\n")
                    return

                if i % 10 == 0:
                    say("\r{}".format(i))

                if i == N-1:
                    self.dropout.set_value(0.0)
                    dev_preds = self.evaluate(eval_func, dev, batch_size, unroll_size)
                    dev_loss = evaluate_average(
                            predictions = dev_preds,
                            masks = None
                        )
                    dev_ppl = np.exp(dev_loss)
                    self.dropout.set_value(dropout_prob)

                    say("\r\n")
                    say( ( "Epoch={}  lr={:.3f}  train_loss={:.3f}  train_ppl={:.1f}  " \
                        +"dev_loss={:.3f}  dev_ppl={:.1f}\t|g|={:.3f}\t[{:.1f}m]\n" ).format(
                            epoch,
                            float(lr.get_value(borrow=True)),
                            train_loss/N,
                            np.exp(train_loss/N),
                            dev_loss,
                            dev_ppl,
                            float(grad_norm),
                            (time.time()-start_time)/60.0
                        ))
                    say("\tp_norm: {}\n".format(
                            self.get_pnorm_stat()
                        ))

                    # halve the learning rate
                    #if args["learning"] == "sgd" and dev_ppl > best_dev-1:
                    #    lr.set_value(np.max([lr.get_value()/2.0, np.float32(0.0001)]))

                    if dev_ppl < best_dev:
                        best_dev = dev_ppl
                        if test is None: continue
                        self.dropout.set_value(0.0)
                        test_preds = self.evaluate(eval_func, test, batch_size, unroll_size)
                        test_loss = evaluate_average(
                                predictions = test_preds,
                                masks = None
                            )
                        test_ppl = np.exp(test_loss)
                        self.dropout.set_value(dropout_prob)
                        say("\tbest_dev={:.1f}  test_loss={:.3f}  test_ppl={:.1f}\n".format(
                                best_dev, test_loss, test_ppl))
                    if best_dev > 200: unchanged += 1

        say("\n")
        #say("Best dev_ppl={}\n".format(best_dev))

    def evaluate(self, eval_func, dev, batch_size, unroll_size):
        predictions = [ ]
        init_state = np.zeros((batch_size, self.n_d*2),
                            dtype=theano.config.floatX)
        N = (len(dev[0])-1)/unroll_size + 1
        for i in xrange(N):
            x = dev[0][i*unroll_size:(i+1)*unroll_size]
            y = dev[1][i*unroll_size:(i+1)*unroll_size]
            pred, init_state = eval_func(x, y, init_state)
            predictions.append(pred)
        return predictions

    def get_pnorm_stat(self):
        lst_norms = [ ]
        for layer in self.layers:
            for p in layer.params:
                vals = p.get_value(borrow=True)
                l2 = np.linalg.norm(vals)
                lst_norms.append("{:.1f}".format(l2))
        return lst_norms

def main(args):
    if args["train"]:
        assert args["dev"]
        train = read_corpus(args["train"], args["max_sent"])
        dev = read_corpus(args["dev"], args["max_sent"])
        test = read_corpus(args["test"], args["max_sent"]) if args["test"] else None
        model = SimpleRNN()
        model.ready(args, train)
        model.train(args, train, dev, test)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = ""
        )
    argparser.add_argument("--dev",
            type = str,
            default = ""
        )
    argparser.add_argument("--test",
            type = str,
            default = ""
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 200
        )
    argparser.add_argument("--learning",
            type = str,
            default = "sgd"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 0.01
        )
    argparser.add_argument("--activation", "-act",
            type = str,
            default = "tanh"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 32
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.5
        )
    argparser.add_argument("--max_sent",
            type = int,
            default = -1
        )
    argparser.add_argument("--max_epoch",
            type = int,
            default = 50
        )
    argparser.add_argument("--unroll_size",
            type = int,
            default = 35
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
            default = 0.0
        )
    argparser.add_argument("--gamma",
            type = float,
            default = 0.95
        )
    argparser.add_argument("--eps",
            type = float,
            default = 1e-8
        )
    argparser.add_argument("--decay_lr",
            type = int,
            default = 0
        )
    args = argparser.parse_args()
    args = vars(args)
    print args
    main(args)
