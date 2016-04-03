import sys
import time
import argparse
import gzip
import cPickle as pickle

from prettytable import PrettyTable
import numpy as np
import theano
import theano.tensor as T

from utils import load_embedding_iterator
from nn import get_activation_by_name, create_optimization_updates
from nn import Layer, EmbeddingLayer, LSTM, GRU, RCNN, Dropout, apply_dropout
from nn.evaluation import evaluate_average

import myio
from myio import say
from evaluation import Evaluation

class Model:
    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.bos_id = embedding_layer.vocab_map["<s>"]
        self.eos_id = embedding_layer.vocab_map["</s>"]
        self.weights = weights

    def ready(self):
        args = self.args
        weights = self.weights

        # len(source) * batch
        idxs = self.idxs = T.imatrix()

        # len(target) * batch
        idys = self.idys = T.imatrix()
        idts = idys[:-1]
        idgs = idys[1:]

        dropout = self.dropout = theano.shared(np.float64(args.dropout).astype(
                            theano.config.floatX))

        embedding_layer = self.embedding_layer

        activation = get_activation_by_name(args.activation)
        n_d = self.n_d = args.hidden_dim
        n_e = self.n_e = embedding_layer.n_d
        n_V = self.n_V = embedding_layer.n_V

        if args.layer.lower() == "rcnn":
            LayerType = RCNN
        elif args.layer.lower() == "lstm":
            LayerType = LSTM
        elif args.layer.lower() == "gru":
            LayerType = GRU

        depth = self.depth = args.depth
        layers = self.layers = [ ]
        for i in range(depth*2):
            if LayerType != RCNN:
                feature_layer = LayerType(
                        n_in = n_e if i/2 == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            else:
                feature_layer = LayerType(
                        n_in = n_e if i/2 == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order,
                        mode = args.mode,
                        has_outgate = args.outgate
                    )
            layers.append(feature_layer)

        self.output_layer = output_layer = Layer(
                n_in = n_d,
                n_out = n_V,
                activation = T.nnet.softmax,
            )

        # feature computation starts here

        # (len*batch)*n_e
        xs_flat = embedding_layer.forward(idxs.ravel())
        xs_flat = apply_dropout(xs_flat, dropout)
        if weights is not None:
            xs_w = weights[idxs.ravel()].dimshuffle((0,'x'))
            xs_flat = xs_flat * xs_w
        # len*batch*n_e
        xs = xs_flat.reshape((idxs.shape[0], idxs.shape[1], n_e))

        # (len*batch)*n_e
        xt_flat = embedding_layer.forward(idts.ravel())
        xt_flat = apply_dropout(xt_flat, dropout)
        if weights is not None:
            xt_w = weights[idts.ravel()].dimshuffle((0,'x'))
            xt_flat = xt_flat * xt_w
        # len*batch*n_e
        xt = xt_flat.reshape((idts.shape[0], idts.shape[1], n_e))

        prev_hs = xs
        prev_ht = xt
        for i in range(depth):
            # len*batch*n_d
            hs = layers[i*2].forward_all(prev_hs, return_c=True)
            ht = layers[i*2+1].forward_all(prev_ht, hs[-1])
            hs = hs[:,:,-n_d:]
            ht = ht[:,:,-n_d:]
            prev_hs = hs
            prev_ht = ht
            prev_hs = apply_dropout(hs, dropout)
            prev_ht = apply_dropout(ht, dropout)

        self.p_y_given_x = output_layer.forward(prev_ht.reshape(
                                (xt_flat.shape[0], n_d)
                            ))

        h_final = hs[-1]
        self.scores2 = -(h_final[1:]-h_final[0]).norm(2,axis=1)
        h_final = self.normalize_2d(h_final)
        self.scores = T.dot(h_final[1:], h_final[0])

        # (len*batch)
        nll = T.nnet.categorical_crossentropy(
                        self.p_y_given_x,
                        idgs.ravel()
                    )
        nll = nll.reshape(idgs.shape)
        self.nll = nll
        self.mask = mask = T.cast(T.neq(idgs, self.padding_id), theano.config.floatX)
        nll = T.sum(nll*mask, axis=0)

        #layers.append(embedding_layer)
        layers.append(output_layer)
        params = [ ]
        for l in self.layers:
            params += l.params
        self.params = params
        say("num of parameters: {}\n".format(
            sum(len(x.get_value(borrow=True).ravel()) for x in params)
        ))

        l2_reg = None
        for p in params:
            if l2_reg is None:
                l2_reg = p.norm(2)
            else:
                l2_reg = l2_reg + p.norm(2)
        l2_reg = l2_reg * args.l2_reg
        self.loss = T.mean(nll)
        self.cost = self.loss + l2_reg

    def train(self, ids_corpus, train, dev=None, test=None, heldout=None):
        args = self.args
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)
        batch_size = args.batch_size
        padding_id = self.padding_id
        bos_id = self.bos_id
        eos_id = self.eos_id

        #train_batches = myio.create_batches(ids_corpus, train, batch_size, padding_id, args.loss)

        updates, lr, gnorm = create_optimization_updates(
                cost = self.cost,
                params = self.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]

        train_func = theano.function(
                inputs = [ self.idxs, self.idys ],
                outputs = [ self.cost, self.loss, gnorm ],
                updates = updates
            )

        eval_func = theano.function(
                inputs = [ self.idxs ],
                #outputs = self.scores2
                outputs = self.scores
            )

        nll_func = theano.function(
                inputs = [ self.idxs, self.idys ],
                outputs = [ self.nll, self.mask ]
            )

        say("\tp_norm: {}\n".format(
                self.get_pnorm_stat()
            ))

        result_table = PrettyTable(["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                    ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])

        unchanged = 0
        best_dev = -1
        dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
        test_MAP = test_MRR = test_P1 = test_P5 = 0
        heldout_PPL = -1

        start_time = 0
        max_epoch = args.max_epoch
        for epoch in xrange(max_epoch):
            unchanged += 1
            if unchanged > 8: break

            start_time = time.time()

            train_batches = myio.create_batches(ids_corpus, train, batch_size,
                                    padding_id, bos_id, eos_id, auto_encode=True)
            N =len(train_batches)

            train_cost = 0.0
            train_loss = 0.0
            train_loss2 = 0.0
            for i in xrange(N):
                # get current batch
                t1, b1, t2 = train_batches[i]

                if args.use_title:
                    idxs, idys = myio.create_one_batch(t1, t2, padding_id)
                    cur_cost, cur_loss, grad_norm = train_func(idxs, idys)
                    train_cost += cur_cost
                    train_loss += cur_loss
                    train_loss2 += cur_loss / idys.shape[0]

                if args.use_body:
                    idxs, idys = myio.create_one_batch(b1, t2, padding_id)
                    cur_cost, cur_loss, grad_norm = train_func(idxs, idys)
                    train_cost += cur_cost
                    train_loss += cur_loss
                    train_loss2 += cur_loss / idys.shape[0]

                if i % 10 == 0:
                    say("\r{}/{}".format(i,N))

                if i == N-1:
                    self.dropout.set_value(0.0)

                    if dev is not None:
                        dev_MAP, dev_MRR, dev_P1, dev_P5 = self.evaluate(dev, eval_func)
                    if test is not None:
                        test_MAP, test_MRR, test_P1, test_P5 = self.evaluate(test, eval_func)
                    if heldout is not None:
                        heldout_PPL = self.evaluate_perplexity(heldout, nll_func)

                    if dev_MRR > best_dev:
                        unchanged = 0
                        best_dev = dev_MRR
                        result_table.add_row(
                            [ epoch ] +
                            [ "%.2f" % x for x in [ dev_MAP, dev_MRR, dev_P1, dev_P5 ] +
                                        [ test_MAP, test_MRR, test_P1, test_P5 ] ]
                        )
                        if args.model:
                            self.save_model(args.model+".pkl.gz")

                    dropout_p = np.float64(args.dropout).astype(
                                theano.config.floatX)
                    self.dropout.set_value(dropout_p)

                    say("\r\n\n")
                    say( ( "Epoch {}\tcost={:.3f}\tloss={:.3f} {:.3f}\t" \
                        +"\tMRR={:.2f},{:.2f}\tPPL={:.1f}\t|g|={:.3f}\t[{:.3f}m]\n" ).format(
                            epoch,
                            train_cost / (i+1),
                            train_loss / (i+1),
                            train_loss2 / (i+1),
                            dev_MRR,
                            best_dev,
                            heldout_PPL,
                            float(grad_norm),
                            (time.time()-start_time)/60.0
                    ))
                    say("\tp_norm: {}\n".format(
                            self.get_pnorm_stat()
                        ))

                    say("\n")
                    say("{}".format(result_table))
                    say("\n")


    def get_pnorm_stat(self):
        lst_norms = [ ]
        for p in self.params:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    def normalize_2d(self, x, eps=1e-8):
        # x is batch*d
        # l2 is batch*1
        l2 = x.norm(2,axis=1).dimshuffle((0,'x'))
        return x/(l2+eps)

    def normalize_3d(self, x, eps=1e-8):
        # x is len*batch*d
        # l2 is len*batch*1
        l2 = x.norm(2,axis=2).dimshuffle((0,1,'x'))
        return x/(l2+eps)

    def average_without_padding(self, x, ids, eps=1e-8):
        # len*batch*1
        mask = T.neq(ids, self.padding_id).dimshuffle((0,1,'x'))
        mask = T.cast(mask, theano.config.floatX)
        # batch*d
        s = T.sum(x*mask,axis=0) / (T.sum(mask,axis=0)+eps)
        return s

    def evaluate(self, data, eval_func):
        res = [ ]
        for t, b, labels in data:
            idts, idbs = myio.create_one_batch(t, b, self.padding_id)
            scores = eval_func(idts)
            #assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_labels = labels[ranks]
            res.append(ranked_labels)
        e = Evaluation(res)
        MAP = e.MAP()*100
        MRR = e.MRR()*100
        P1 = e.Precision(1)*100
        P5 = e.Precision(5)*100
        return MAP, MRR, P1, P5

    def evaluate_perplexity(self, data, nll_func):
        nll_preds = [ ]
        nll_masks = [ ]
        for idbs, idts in data:
            nll, mask = nll_func(idbs, idts)
            assert nll.shape == mask.shape
            nll_preds.append(nll)
            nll_masks.append(mask)
        avg_nll = evaluate_average(
                    predictions = nll_preds,
                    masks = nll_masks
                )
        return np.exp(avg_nll)

    def save_model(self, path):
        args = self.args
        lst_params = [ ]
        for i in range(args.depth):
            lst_params.append(self.layers[i*2].params)
        with gzip.open(path,"w") as fout:
            pickle.dump(
                    { "d": args.hidden_dim,
                      "layer_type": args.layer,
                      "args": args,
                      "params": lst_params },
                    fout,
                    protocol = pickle.HIGHEST_PROTOCOL
                )
        say(" \tmodel saved.\n")


def main(args):
    raw_corpus = myio.read_corpus(args.corpus)
    embedding_layer = myio.create_embedding_layer(
                raw_corpus,
                n_d = args.hidden_dim,
                cut_off = args.cut_off,
                embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
            )
    ids_corpus = myio.map_corpus(raw_corpus, embedding_layer)
    say("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]
    bos_id = embedding_layer.vocab_map["<s>"]
    eos_id = embedding_layer.vocab_map["</s>"]

    if args.reweight:
        weights = myio.create_idf_weights(args.corpus, embedding_layer)

    if args.dev:
        dev = myio.read_annotations(args.dev, K_neg=20, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus, dev, padding_id)
    if args.test:
        test = myio.read_annotations(args.test, K_neg=20, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus, test, padding_id)

    if args.heldout:
        with open(args.heldout) as fin:
            heldout_ids = fin.read().split()
        heldout_corpus = dict((id, ids_corpus[id]) for id in heldout_ids if id in ids_corpus)
        train_corpus = dict((id, ids_corpus[id]) for id in ids_corpus
                                                if id not in heldout_corpus)
        heldout = myio.create_batches(heldout_corpus, [ ], args.batch_size,
                    padding_id, bos_id, eos_id, auto_encode=True)
        heldout = [ myio.create_one_batch(b1, t2, padding_id) for t1, b1, t2 in heldout ]
        say("heldout examples={}\n".format(len(heldout_corpus)))

    if args.train:
        model = Model(args, embedding_layer,
                      weights=weights if args.reweight else None)

        start_time = time.time()
        train = myio.read_annotations(args.train)
        if not args.use_anno: train = [ ]
        train_batches = myio.create_batches(ids_corpus, train, args.batch_size,
                    model.padding_id, model.bos_id, model.eos_id, auto_encode=True)
        say("{} to create batches\n".format(time.time()-start_time))

        model.ready()
        model.train(
                ids_corpus if not args.heldout else train_corpus,
                train,
                dev if args.dev else None,
                test if args.test else None,
                heldout if args.heldout else None
            )

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus",
            type = str
        )
    argparser.add_argument("--train",
            type = str,
            default = ""
        )
    argparser.add_argument("--test",
            type = str,
            default = ""
        )
    argparser.add_argument("--dev",
            type = str,
            default = ""
        )
    argparser.add_argument("--heldout",
            type = str,
            default = ""
        )
    argparser.add_argument("--embeddings",
            type = str,
            default = ""
        )
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 200
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adam"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 0.001
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.00001
        )
    argparser.add_argument("--activation", "-act",
            type = str,
            default = "tanh"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 256
        )
    argparser.add_argument("--depth",
            type = int,
            default = 1
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.0
        )
    argparser.add_argument("--max_epoch",
            type = int,
            default = 50
        )
    argparser.add_argument("--cut_off",
            type = int,
            default = 1
        )
    argparser.add_argument("--max_seq_len",
            type = int,
            default = 100
        )
    argparser.add_argument("--normalize",
            type = int,
            default = 1
        )
    argparser.add_argument("--reweight",
            type = int,
            default = 1
        )
    argparser.add_argument("--order",
            type = int,
            default = 2
        )
    argparser.add_argument("--layer",
            type = str,
            default = "rcnn"
        )
    argparser.add_argument("--mode",
            type = int,
            default = 1
        )
    argparser.add_argument("--outgate",
            type = int,
            default = 0
        )
    argparser.add_argument("--model",
            type = str,
            default = ""
        )
    argparser.add_argument("--use_title",
            type = int,
            default = 1
        )
    argparser.add_argument("--use_body",
            type = int,
            default = 1
        )
    argparser.add_argument("--use_anno",
            type = int,
            default = 1
        )
    args = argparser.parse_args()
    assert args.use_title or args.use_body
    print args
    print ""
    main(args)
