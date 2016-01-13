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
from nn import EmbeddingLayer, LSTM, GRU, RCNN, Dropout, apply_dropout
import myio
from myio import say
from evaluation import Evaluation

class Model:
    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights

    def ready(self):
        args = self.args
        weights = self.weights

        # len(title) * batch
        idts = self.idts = T.imatrix()

        # len(body) * batch
        idbs = self.idbs = T.imatrix()

        # num pairs * 3, or num queries * candidate size
        idps = self.idps = T.imatrix()

        dropout = self.dropout = theano.shared(np.float64(args.dropout).astype(
                            theano.config.floatX))
        dropout_op = self.dropout_op = Dropout(self.dropout)

        embedding_layer = self.embedding_layer

        activation = get_activation_by_name(args.activation)
        n_d = self.n_d = args.hidden_dim
        n_e = self.n_e = embedding_layer.n_d

        if args.layer.lower() == "rcnn":
            LayerType = RCNN
        elif args.layer.lower() == "lstm":
            LayerType = LSTM
        elif args.layer.lower() == "gru":
            LayerType = GRU

        depth = self.depth = args.depth
        layers = self.layers = [ ]
        for i in range(depth):
            if LayerType != RCNN:
                feature_layer = LayerType(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            else:
                feature_layer = LayerType(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order,
                        mode = args.mode,
                        has_outgate = args.outgate
                    )
            layers.append(feature_layer)

        # feature computation starts here

        # (len*batch)*n_e
        xt = embedding_layer.forward(idts.ravel())
        if weights is not None:
            xt_w = weights[idts.ravel()].dimshuffle((0,'x'))
            xt = xt * xt_w

        # len*batch*n_e
        xt = xt.reshape((idts.shape[0], idts.shape[1], n_e))
        xt = apply_dropout(xt, dropout)

        # (len*batch)*n_e
        xb = embedding_layer.forward(idbs.ravel())
        if weights is not None:
            xb_w = weights[idbs.ravel()].dimshuffle((0,'x'))
            xb = xb * xb_w

        # len*batch*n_e
        xb = xb.reshape((idbs.shape[0], idbs.shape[1], n_e))
        xb = apply_dropout(xb, dropout)

        prev_ht = xt
        prev_hb = xb
        for i in range(depth):
            # len*batch*n_d
            ht = layers[i].forward_all(prev_ht)
            hb = layers[i].forward_all(prev_hb)
            prev_ht = ht
            prev_hb = hb

        # normalize vectors
        if args.normalize:
            ht = self.normalize_3d(ht)
            hb = self.normalize_3d(hb)
            say("h_title dtype: {}\n".format(ht.dtype))

        # average over length, ignore paddings
        # batch * d
        if args.average:
            ht = self.average_without_padding(ht, idts)
            hb = self.average_without_padding(hb, idbs)
        else:
            ht = ht[-1]
            hb = hb[-1]
        say("h_avg_title dtype: {}\n".format(ht.dtype))

        # batch * d
        h_final = (ht+hb)*0.5
        h_final = apply_dropout(h_final, dropout)
        h_final = self.normalize_2d(h_final)
        say("h_final dtype: {}\n".format(ht.dtype))

        # For testing:
        #   first one in batch is query, the rest are candidate questions
        self.scores = T.dot(h_final[1:], h_final[0])

        # For training:
        xp = h_final[idps.ravel()]
        xp = xp.reshape((idps.shape[0], idps.shape[1], n_d))
        # num query * n_d
        query_vecs = xp[:,0,:]
        # num query
        pos_scores = T.sum(query_vecs*xp[:,1,:], axis=1)
        # num query * candidate size
        neg_scores = T.sum(query_vecs.dimshuffle((0,'x',1))*xp[:,2:,:], axis=2)
        # num query
        neg_scores = T.max(neg_scores, axis=1)
        diff = neg_scores - pos_scores + 1.0
        loss = T.mean( (diff>0)*diff )
        self.loss = loss

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
        self.cost = self.loss + l2_reg

    def train(self, ids_corpus, train, dev=None, test=None):
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)
        batch_size = args.batch_size
        padding_id = self.padding_id

        #train_batches = myio.create_batches(ids_corpus, train, batch_size, padding_id)

        updates, lr, gnorm = create_optimization_updates(
                cost = self.cost,
                params = self.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]

        train_func = theano.function(
                inputs = [ self.idts, self.idbs, self.idps ],
                outputs = [ self.cost, self.loss, gnorm ],
                updates = updates
            )

        eval_func = theano.function(
                inputs = [ self.idts, self.idbs ],
                outputs = self.scores
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
        start_time = 0
        max_epoch = args.max_epoch
        for epoch in xrange(max_epoch):
            unchanged += 1
            if unchanged > 15: break

            start_time = time.time()

            train = myio.read_annotations(args.train)
            train_batches = myio.create_batches(ids_corpus, train, batch_size,
                                    padding_id, pad_left = not args.average)
            N =len(train_batches)

            train_loss = 0.0
            train_cost = 0.0

            for i in xrange(N):
                # get current batch
                idts, idbs, idps = train_batches[i]

                cur_cost, cur_loss, grad_norm = train_func(idts, idbs, idps)
                train_loss += cur_loss
                train_cost += cur_cost

                if i % 10 == 0:
                    say("\r{}/{}".format(i,N))

                if i == N-1:
                    self.dropout.set_value(0.0)

                    if dev is not None:
                        dev_MAP, dev_MRR, dev_P1, dev_P5 = self.evaluate(dev, eval_func)
                    if test is not None:
                        test_MAP, test_MRR, test_P1, test_P5 = self.evaluate(test, eval_func)

                    if dev_MRR > best_dev:
                        unchanged = 0
                        best_dev = dev_MRR
                        result_table.add_row(
                            [ epoch ] +
                            [ "%.2f" % x for x in [ dev_MAP, dev_MRR, dev_P1, dev_P5 ] +
                                        [ test_MAP, test_MRR, test_P1, test_P5 ] ]
                        )

                    dropout_p = np.float64(args.dropout).astype(
                                theano.config.floatX)
                    self.dropout.set_value(dropout_p)

                    say("\r\n\n")
                    say( ( "Epoch {}\tcost={:.3f}\tloss={:.3f}" \
                        +"\tMRR={:.2f},{:.2f}\t|g|={:.3f}\t[{:.3f}m]\n" ).format(
                            epoch,
                            train_cost / (i+1),
                            train_loss / (i+1),
                            dev_MRR,
                            best_dev,
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
        for idts, idbs, labels in data:
            scores = eval_func(idts, idbs)
            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_labels = labels[ranks]
            res.append(ranked_labels)
        e = Evaluation(res)
        MAP = e.MAP()*100
        MRR = e.MRR()*100
        P1 = e.Precision(1)*100
        P5 = e.Precision(5)*100
        return MAP, MRR, P1, P5

    def load_pretrained_parameters(self, args):
        with gzip.open(args.load_pretrain) as fin:
            data = pickle.load(fin)
            assert args.hidden_dim == data["d"]
            #assert args.layer == data["layer_type"]
            for l, p in zip(self.layers, data["params"]):
                l.params = p

def main(args):
    raw_corpus = myio.read_corpus(args.corpus)
    embedding_layer = myio.create_embedding_layer(
                raw_corpus,
                n_d = args.hidden_dim,
                cut_off = args.cut_off,
                embs = load_embedding_iterator(args.embeddings) if args.embeddings else None
            )
    ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, max_len=args.max_seq_len)
    say("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.reweight:
        weights = myio.create_idf_weights(args.corpus, embedding_layer)

    if args.dev:
        dev = myio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus, dev, padding_id, pad_left = not args.average)
    if args.test:
        test = myio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus, test, padding_id, pad_left = not args.average)

    if args.train:
        start_time = time.time()
        train = myio.read_annotations(args.train)
        train_batches = myio.create_batches(ids_corpus, train, args.batch_size,
                                padding_id, pad_left = not args.average)
        say("{} to create batches\n".format(time.time()-start_time))
        say("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
                sum(len(x[2].ravel()) for x in train_batches)
            ))
        train_batches = None

        model = Model(args, embedding_layer,
                      weights=weights if args.reweight else None)
        model.ready()

        # set parameters using pre-trained network
        if args.load_pretrain:
            model.load_pretrained_parameters(args)

        model.train(
                ids_corpus,
                train,
                dev if args.dev else None,
                test if args.test else None
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
            default = 1e-7
        )
    argparser.add_argument("--activation", "-act",
            type = str,
            default = "tanh"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 40
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
    argparser.add_argument("--load_pretrain",
            type = str,
            default = ""
        )
    argparser.add_argument("--average",
            type = int,
            default = 0
        )
    args = argparser.parse_args()
    print args
    print ""
    main(args)
