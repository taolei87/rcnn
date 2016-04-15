import os, sys, random, argparse, time, math, gzip
import cPickle as pickle
from collections import Counter

import numpy as np
import theano
import theano.tensor as T

from nn import get_activation_by_name, create_optimization_updates, softmax
from nn import Layer, EmbeddingLayer, LSTM, RCNN, StrCNN, Dropout, apply_dropout
from utils import say, load_embedding_iterator

np.set_printoptions(precision=3)

def read_corpus(path):
    with open(path) as fin:
        lines = fin.readlines()
    lines = [ x.strip().split() for x in lines ]
    lines = [ x for x in lines if x ]
    corpus_x = [ x[1:] for x in lines ]
    corpus_y = [ int(x[0]) for x in lines ]
    return corpus_x, corpus_y

def create_one_batch(ids, x, y):
    batch_x = np.column_stack( [ x[i] for i in ids ] )
    batch_y = np.array( [ y[i] for i in ids ] )
    return batch_x, batch_y

# shuffle training examples and create mini-batches
def create_batches(perm, x, y, batch_size):

    # sort sequences based on their length
    # permutation is necessary if we want different batches every epoch
    lst = sorted(perm, key=lambda i: len(x[i]))

    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i]) == len(x[ids[0]]):
            ids.append(i)
        else:
            bx, by = create_one_batch(ids, x, y)
            batches_x.append(bx)
            batches_y.append(by)
            ids = [ i ]
    bx, by = create_one_batch(ids, x, y)
    batches_x.append(bx)
    batches_y.append(by)

    # shuffle batches
    batch_perm = range(len(batches_x))
    random.shuffle(batch_perm)
    batches_x = [ batches_x[i] for i in batch_perm ]
    batches_y = [ batches_y[i] for i in batch_perm ]
    return batches_x, batches_y



class Model:
    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        args = self.args
        embedding_layer = self.embedding_layer
        self.n_hidden = args.hidden_dim
        self.n_in = embedding_layer.n_d
        dropout = self.dropout = theano.shared(
                np.float64(args.dropout_rate).astype(theano.config.floatX)
            )

        # x is length * batch_size
        # y is batch_size
        self.x = T.imatrix('x')
        self.y = T.ivector('y')

        x = self.x
        y = self.y
        n_hidden = self.n_hidden
        n_in = self.n_in

        # fetch word embeddings
        # (len * batch_size) * n_in
        slices  = embedding_layer.forward(x.ravel())
        self.slices = slices

        # 3-d tensor, len * batch_size * n_in
        slices = slices.reshape( (x.shape[0], x.shape[1], n_in) )

        # stacking the feature extraction layers
        pooling = args.pooling
        depth = args.depth
        layers = self.layers = [ ]
        prev_output = slices
        prev_output = apply_dropout(prev_output, dropout, v2=True)
        size = 0
        softmax_inputs = [ ]
        activation = get_activation_by_name(args.act)
        for i in range(depth):
            if args.layer.lower() == "lstm":
                layer = LSTM(
                            n_in = n_hidden if i > 0 else n_in,
                            n_out = n_hidden
                        )
            elif args.layer.lower() == "strcnn":
                layer = StrCNN(
                            n_in = n_hidden if i > 0 else n_in,
                            n_out = n_hidden,
                            activation = activation,
                            decay = args.decay,
                            order = args.order
                        )
            elif args.layer.lower() == "rcnn":
                layer = RCNN(
                            n_in = n_hidden if i > 0 else n_in,
                            n_out = n_hidden,
                            activation = activation,
                            order = args.order,
                            mode = args.mode
                        )
            else:
                raise Exception("unknown layer type: {}".format(args.layer))

            layers.append(layer)
            prev_output = layer.forward_all(prev_output)
            if pooling:
                softmax_inputs.append(T.sum(prev_output, axis=0)) # summing over columns
            else:
                softmax_inputs.append(prev_output[-1])
            prev_output = apply_dropout(prev_output, dropout)
            size += n_hidden

        # final feature representation is the concatenation of all extraction layers
        if pooling:
            softmax_input = T.concatenate(softmax_inputs, axis=1) / x.shape[0]
        else:
            softmax_input = T.concatenate(softmax_inputs, axis=1)
        softmax_input = apply_dropout(softmax_input, dropout, v2=True)

        # feed the feature repr. to the softmax output layer
        layers.append( Layer(
                n_in = size,
                n_out = self.nclasses,
                activation = softmax,
                has_bias = False
        ) )

        for l,i in zip(layers, range(len(layers))):
            say("layer {}: n_in={}\tn_out={}\n".format(
                i, l.n_in, l.n_out
            ))

        # unnormalized score of y given x
        self.p_y_given_x = layers[-1].forward(softmax_input)
        self.pred = T.argmax(self.p_y_given_x, axis=1)
        self.nll_loss = T.mean( T.nnet.categorical_crossentropy(
                                    self.p_y_given_x,
                                    y
                            ))

        # adding regularizations
        self.l2_sqr = None
        self.params = [ ]
        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p**2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p**2)

        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                        for x in self.params)
        say("total # parameters: {}\n".format(nparams))


    def save_model(self, path, args):
         # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.params ], args, self.nclasses),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            param_values, args, nclasses = pickle.load(fin)

        self.args = args
        self.nclasses = nclasses
        self.ready()
        for x,v in zip(self.params, param_values):
            x.set_value(v)

    def eval_accuracy(self, preds, golds):
        fine = sum([ sum(p == y) for p,y in zip(preds, golds) ]) + 0.0
        fine_tot = sum( [ len(y) for y in golds ] )
        return fine/fine_tot


    def train(self, train, dev, test):
        args = self.args
        trainx, trainy = train
        batch_size = args.batch

        if dev:
            dev_batches_x, dev_batches_y = create_batches(
                    range(len(dev[0])),
                    dev[0],
                    dev[1],
                    batch_size
            )

        if test:
            test_batches_x, test_batches_y = create_batches(
                    range(len(test[0])),
                    test[0],
                    test[1],
                    batch_size
            )

        cost = self.nll_loss + self.l2_sqr

        updates, lr, gnorm = create_optimization_updates(
                cost = cost,
                params = self.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]

        train_model = theano.function(
             inputs = [self.x, self.y],
             outputs = [ cost, gnorm ],
             updates = updates,
             allow_input_downcast = True
        )

        eval_acc = theano.function(
             inputs = [self.x],
             outputs = self.pred,
             allow_input_downcast = True
        )

        unchanged = 0
        best_dev = 0.0
        dropout_prob = np.float64(args.dropout_rate).astype(theano.config.floatX)

        start_time = time.time()
        eval_period = args.eval_period

        perm = range(len(trainx))

        say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")
        for epoch in xrange(args.max_epochs):
            unchanged += 1
            if unchanged > 20: return
            train_loss = 0.0

            random.shuffle(perm)
            batches_x, batches_y = create_batches(perm, trainx, trainy, batch_size)

            N = len(batches_x)
            for i in xrange(N):

                if i % 100 == 0:
                    sys.stdout.write("\r%d" % i)
                    sys.stdout.flush()

                x = batches_x[i]
                y = batches_y[i]

                va, grad_norm = train_model(x, y)
                train_loss += va

                # debug
                if math.isnan(va):
                    print ""
                    print i-1, i
                    print x
                    print y
                    return

                if (i == N-1) or (eval_period > 0 and (i+1) % eval_period == 0):
                    self.dropout.set_value(0.0)

                    say( "\n" )
                    say( "Epoch %.1f\tloss=%.4f\t|g|=%s  [%.2fm]\n" % (
                            epoch + (i+1)/(N+0.0),
                            train_loss / (i+1),
                            float(grad_norm),
                            (time.time()-start_time) / 60.0
                    ))
                    say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")

                    if dev:
                        preds = [ eval_acc(x) for x in dev_batches_x ]
                        nowf_dev = self.eval_accuracy(preds, dev_batches_y)
                        if nowf_dev > best_dev:
                            unchanged = 0
                            best_dev = nowf_dev
                            if args.save:
                                self.save_model(args.save, args)

                        say("\tdev accuracy=%.4f\tbest=%.4f\n" % (
                                nowf_dev,
                                best_dev
                        ))
                        if args.test and nowf_dev == best_dev:
                            preds = [ eval_acc(x) for x in test_batches_x ]
                            nowf_test = self.eval_accuracy(preds, test_batches_y)
                            say("\ttest accuracy=%.4f\n" % (
                                    nowf_test,
                            ))

                        if best_dev > nowf_dev + 0.05:
                            return

                    self.dropout.set_value(dropout_prob)

                    start_time = time.time()

    def evaluate_batches(self, batches_x, batches_y, eval_function):
        preds = [ eval_function(x) for x in batches_x ]
        return self.eval_accuracy(preds, batches_y)

    def evaluate_set(self, data_x, data_y):

        args = self.args

        # compile prediction function
        eval_acc = theano.function(
             inputs = [self.x],
             outputs = self.pred,
             allow_input_downcast = True
        )

        # create batches by grouping sentences of the same length together
        batches_x, batches_y = create_batches(
                    range(len(data_x)),
                    data_x,
                    data_y,
                    args.batch
            )

        # evaluate on the data set
        dropout_prob = np.float64(args.dropout_rate).astype(theano.config.floatX)
        self.dropout.set_value(0.0)
        accuracy = self.evaluate_batches(batches_x, batches_y, eval_acc)
        self.dropout.set_value(dropout_prob)
        return accuracy


def main(args):
    print args

    model = None

    assert args.embedding, "Pre-trained word embeddings required."

    embedding_layer = EmbeddingLayer(
                n_d = args.hidden_dim,
                vocab = [ "<unk>" ],
                embs = load_embedding_iterator(args.embedding)
            )

    if args.train:
        train_x, train_y = read_corpus(args.train)
        train_x = [ embedding_layer.map_to_ids(x) for x in train_x ]

    if args.dev:
        dev_x, dev_y = read_corpus(args.dev)
        dev_x = [ embedding_layer.map_to_ids(x) for x in dev_x ]

    if args.test:
        test_x, test_y = read_corpus(args.test)
        test_x = [ embedding_layer.map_to_ids(x) for x in test_x ]

    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = max(train_y)+1
            )
        model.ready()
        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                (test_x, test_y) if args.test else None,
            )

    if args.load and args.test and not args.train:
        # model.args and model.nclasses will be loaded from file
        model = Model(
                    args = None,
                    embedding_layer = embedding_layer,
                    nclasses = -1
            )
        model.load_model(args.load)
        accuracy = model.evaluate_set(test_x, test_y)
        print accuracy


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data"
        )
    argparser.add_argument("--dev",
            type = str,
            default = "",
            help = "path to development data"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 200,
            help = "hidden dimensions"
        )
    argparser.add_argument("--decay",
            type = float,
            default = 0.5,
            help = "the decay factor of StrCNN layer"
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
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--eval_period",
            type = int,
            default = -1,
            help = "evaluate on dev every period"
        )
    argparser.add_argument("--dropout_rate",
            type = float,
            default = 0.3,
            help = "dropout probability"
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.00001
        )
    argparser.add_argument("--embedding",
            type = str,
            default = ""
        )
    argparser.add_argument("--batch",
            type = int,
            default = 16,
            help = "mini-batch size"
        )
    argparser.add_argument("--depth",
            type = int,
            default = 2,
            help = "number of feature extraction layers (min:1)"
        )
    argparser.add_argument("--order",
            type = int,
            default = 2,
            help = "when the order is k, we use up tp k-grams"
        )
    argparser.add_argument("--act",
            type = str,
            default = "relu",
            help = "activation function (none, relu, tanh, etc.)"
        )
    argparser.add_argument("--layer",
            type = str,
            default = "strcnn",
            help = "type of neural net (LSTM, RCNN, StrCNN)"
        )
    argparser.add_argument("--mode",
            type = int,
            default = 1
        )
    argparser.add_argument("--save",
            type = str,
            default = "",
            help = "save model to this file"
        )
    argparser.add_argument("--load",
            type = str,
            default = "",
            help = "load model from this file"
        )
    argparser.add_argument("--pooling",
            type = int,
            default = 1,
            help = "whether to use mean pooling or take the last vector"
        )
    args = argparser.parse_args()
    main(args)


