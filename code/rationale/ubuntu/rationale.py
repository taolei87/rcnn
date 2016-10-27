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
from nn import EmbeddingLayer, LSTM, GRU, RCNN, apply_dropout
import myio
from myio import say
from evaluation import Evaluation

from extended_layers import ExtRCNN, ExtLSTM, ZLayer
import options

class Generator(object):
    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights

    def ready(self):
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = self.padding_id
        weights = self.weights

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # len*batch
        x = self.x = T.imatrix()

        n_d = args.hidden_dim2
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]
        layer_type = args.layer.lower()
        for i in xrange(2):
            if layer_type == "rcnn":
                l = RCNN(
                        n_in = n_e,# if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order
                    )
            elif layer_type == "lstm":
                l = LSTM(
                        n_in = n_e,# if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            layers.append(l)

        # len * batch
        masks = T.cast(T.neq(x, padding_id), "float32")

        #masks = masks.dimshuffle((0,1,"x"))

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        if weights is not None:
            embs_w = weights[x.ravel()].dimshuffle((0,'x'))
            embs = embs * embs_w

        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)
        self.word_embs = embs
        flipped_embs = embs[::-1]

        # len*bacth*n_d
        h1 = layers[0].forward_all(embs)
        h2 = layers[1].forward_all(flipped_embs)
        h_final = T.concatenate([h1, h2[::-1]], axis=2)
        h_final = apply_dropout(h_final, dropout)
        size = n_d * 2

        output_layer = self.output_layer = ZLayer(
                n_in = size,
                n_hidden = n_d,
                activation = activation
            )

        # sample z given text (i.e. x)
        z_pred, sample_updates = output_layer.sample_all(h_final)

        # we are computing approximated gradient by sampling z;
        # so should mark sampled z not part of the gradient propagation path
        #
        z_pred = self.z_pred = theano.gradient.disconnected_grad(z_pred)
        self.sample_updates = sample_updates
        print "z_pred", z_pred.ndim

        self.p1 = T.sum(masks*z_pred) / (T.sum(masks) + 1e-8)

        # len*batch*1
        probs = output_layer.forward_all(h_final, z_pred)
        print "probs", probs.ndim

        logpz = - T.nnet.binary_crossentropy(probs, z_pred) * masks
        logpz = self.logpz = logpz.reshape(x.shape)
        probs = self.probs = probs.reshape(x.shape)

        # batch
        z = z_pred
        self.zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
        self.zdiff = T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)

        params = self.params = [ ]
        for l in layers + [ output_layer ]:
            for p in l.params:
                params.append(p)
        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                                        for x in params)
        say("total # parameters: {}\n".format(nparams))

        l2_cost = None
        for p in params:
            if l2_cost is None:
                l2_cost = T.sum(p**2)
            else:
                l2_cost = l2_cost + T.sum(p**2)
        l2_cost = l2_cost * args.l2_reg
        self.l2_cost = l2_cost


class Encoder:
    def __init__(self, args, embedding_layer, generator, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.generator = generator

    def ready(self):
        generator = self.generator
        args = self.args
        weights = self.weights

        dropout = generator.dropout

        # len(text) * batch
        idts = generator.x
        z = generator.z_pred
        z = z.dimshuffle((0,1,"x"))

        # batch * 2
        pairs = self.pairs = T.imatrix()

        # num pairs * 3, or num queries * candidate size
        triples = self.triples = T.imatrix()

        embedding_layer = self.embedding_layer

        activation = get_activation_by_name(args.activation)
        n_d = self.n_d = args.hidden_dim
        n_e = self.n_e = embedding_layer.n_d

        if args.layer.lower() == "rcnn":
            LayerType = RCNN
            LayerType2 = ExtRCNN
        elif args.layer.lower() == "lstm":
            LayerType = LSTM
            LayerType2 = ExtLSTM
        #elif args.layer.lower() == "gru":
        #    LayerType = GRU

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

        extlayers = self.extlayers = [ ]
        for i in range(depth):
            if LayerType != RCNN:
                feature_layer = LayerType2(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation
                    )
            else:
                feature_layer = LayerType2(
                        n_in = n_e if i == 0 else n_d,
                        n_out = n_d,
                        activation = activation,
                        order = args.order,
                        mode = args.mode,
                        has_outgate = args.outgate
                    )
            feature_layer.copy_params(layers[i])
            extlayers.append(feature_layer)


        # feature computation starts here

        xt = generator.word_embs

        # encode full text into representation
        prev_ht = self.xt = xt
        for i in range(depth):
            # len*batch*n_d
            ht = layers[i].forward_all(prev_ht)
            prev_ht = ht

        # encode selected text into representation
        prev_htz = self.xt = xt
        for i in range(depth):
            # len*batch*n_d
            htz = extlayers[i].forward_all(prev_htz, z)
            prev_htz = htz

        # normalize vectors
        if args.normalize:
            ht = self.normalize_3d(ht)
            htz = self.normalize_3d(htz)
            say("h_title dtype: {}\n".format(ht.dtype))

        self.ht = ht
        self.htz = htz

        # average over length, ignore paddings
        # batch * d
        if args.average:
            ht = self.average_without_padding(ht, idts)
            htz = self.average_without_padding(htz, idts, z)
        else:
            ht = ht[-1]
            htz = htz[-1]
        say("h_avg_title dtype: {}\n".format(ht.dtype))

        # batch * d
        h_final = apply_dropout(ht, dropout)
        h_final = self.normalize_2d(h_final)
        hz_final = apply_dropout(htz, dropout)
        hz_final = self.normalize_2d(hz_final)
        self.h_final = h_final
        self.hz_final = hz_final

        say("h_final dtype: {}\n".format(ht.dtype))

        # For testing:
        #   first one in batch is query, the rest are candidate questions
        self.scores = T.dot(h_final[1:], h_final[0])
        self.scores_z = T.dot(hz_final[1:], hz_final[0])

        # For training encoder:
        xp = h_final[triples.ravel()]
        xp = xp.reshape((triples.shape[0], triples.shape[1], n_d))
        # num query * n_d
        query_vecs = xp[:,0,:]
        # num query
        pos_scores = T.sum(query_vecs*xp[:,1,:], axis=1)
        # num query * candidate size
        neg_scores = T.sum(query_vecs.dimshuffle((0,'x',1))*xp[:,2:,:], axis=2)
        # num query
        neg_scores = T.max(neg_scores, axis=1)
        diff = neg_scores - pos_scores + 1.0
        hinge_loss = T.mean( (diff>0)*diff )

        # For training generator

        # batch
        self_cosine_distance = 1.0 - T.sum(hz_final * h_final, axis=1)
        pair_cosine_distance = 1.0 - T.sum(hz_final * h_final[pairs[:,1]], axis=1)
        alpha = args.alpha
        loss_vec = self_cosine_distance*alpha + pair_cosine_distance*(1-alpha)
        #loss_vec = self_cosine_distance*0.2 + pair_cosine_distance*0.8

        zsum = generator.zsum
        zdiff = generator.zdiff
        logpz = generator.logpz

        sfactor = args.sparsity
        cfactor = args.sparsity * args.coherent
        scost_vec = zsum*sfactor + zdiff*cfactor

        # batch
        cost_vec = loss_vec + scost_vec
        cost_logpz = T.mean(cost_vec * T.sum(logpz, axis=0))
        loss = self.loss = T.mean(loss_vec)
        sparsity_cost = self.sparsity_cost = T.mean(scost_vec)
        self.obj =  loss + sparsity_cost

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
                l2_reg = T.sum(p**2) #p.norm(2)
            else:
                l2_reg = l2_reg + T.sum(p**2) #p.norm(2)
        l2_reg = l2_reg * args.l2_reg
        self.l2_cost = l2_reg

        beta = args.beta
        self.cost_g = cost_logpz + generator.l2_cost
        self.cost_e = hinge_loss + loss*beta + l2_reg
        print "cost dtype", self.cost_g.dtype, self.cost_e.dtype

    def normalize_2d(self, x, eps=1e-6):
        # x is batch*d
        # l2 is batch*1
        #l2 = x.norm(2,axis=1).dimshuffle((0,'x'))
        l2 = T.sum(x**2, axis=1).dimshuffle((0,"x"))
        return x/T.sqrt(l2+eps)

    def normalize_3d(self, x, eps=1e-8):
        # x is len*batch*d
        # l2 is len*batch*1
        #l2 = x.norm(2,axis=2).dimshuffle((0,1,'x'))
        l2 = T.sum(x**2, axis=2).dimshuffle((0,1,"x"))
        return x/T.sqrt(l2+eps)

    def average_without_padding(self, x, ids, z=None, eps=1e-8):
        # len*batch*1
        mask = T.neq(ids, self.padding_id).dimshuffle((0,1,'x'))
        if z is not None: mask = mask * z
        mask = T.cast(mask, theano.config.floatX)
        # batch*d
        s = T.sum(x*mask,axis=0) / (T.sum(mask,axis=0)+eps)
        return s

    def load_pretrained_parameters(self, args):
        with gzip.open(args.load_pretrain) as fin:
            data = pickle.load(fin)
            assert args.hidden_dim == data["d"]
            #assert args.layer == data["layer_type"]
            for l, p in zip(self.layers, data["params"]):
                l.params = p

    def save_model(self, path):
        if not path.endswith(".pkl.gz"):
            path = path + (".gz" if path.endswith(".pkl") else ".pkl.gz")

        args = self.args
        params = [ x.params for x in self.layers ]
        weights = self.weights
        with gzip.open(path, "w") as fout:
            pickle.dump(
                {
                    "args": args,
                    "d"   : args.hidden_dim,
                    "params": params,
                    "weights": weights
                },
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        with gzip.open(path) as fin:
            data = pickle.load(fin)
        return data

    def set_model(self, data):
        self.args = data["args"]
        self.weights = data["weights"]
        self.ready()
        for l, p in zip(self.layers, data["params"]):
            l.params = p

class Model(object):

    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.generator = Generator(args, embedding_layer, weights)
        self.encoder = Encoder(args, embedding_layer, self.generator, weights)
        self.padding_id = self.encoder.padding_id

    def ready(self):
        self.generator.ready()
        self.encoder.ready()
        self.x = self.generator.x
        self.triples = self.encoder.triples
        self.pairs = self.encoder.pairs
        self.z = self.generator.z_pred
        self.dropout = self.generator.dropout
        self.params = self.encoder.params + self.generator.params

    def train(self, ids_corpus, train, dev=None, test=None):
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)
        batch_size = args.batch_size
        padding_id = self.padding_id

        #train_batches = myio.create_batches(ids_corpus, train, batch_size, padding_id)

        if dev is not None:
            dev, dev_raw = dev
        if test is not None:
            test, test_raw = test

        if args.joint:
            updates_e, lr_e, gnorm_e = create_optimization_updates(
                    cost = self.encoder.cost_e, #self.encoder.cost,
                    params = self.encoder.params,
                    lr = args.learning_rate*0.1,
                    method = args.learning
                )[:3]
        else:
            updates_e = {}

        updates_g, lr_g, gnorm_g = create_optimization_updates(
                cost = self.encoder.cost_g,
                params = self.generator.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]

        train_func = theano.function(
                inputs = [ self.x, self.triples, self.pairs ],
                outputs = [ self.encoder.obj, self.encoder.loss, \
                        self.encoder.sparsity_cost, self.generator.p1, gnorm_g ],
                updates = updates_g.items() + updates_e.items() + self.generator.sample_updates,
                #no_default_updates = True,
                on_unused_input= "ignore"
            )

        eval_func = theano.function(
                inputs = [ self.x ],
                outputs = self.encoder.scores
            )

        eval_func2 = theano.function(
                inputs = [ self.x ],
                outputs = [ self.encoder.scores_z, self.generator.p1, self.z ],
                updates = self.generator.sample_updates,
                #no_default_updates = True
            )


        say("\tp_norm: {}\n".format(
                self.get_pnorm_stat(self.encoder.params)
            ))
        say("\tp_norm: {}\n".format(
                self.get_pnorm_stat(self.generator.params)
            ))

        result_table = PrettyTable(["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                    ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])
        last_train_avg_cost = None
        tolerance = 0.5 + 1e-3
        unchanged = 0
        best_dev = -1
        dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
        test_MAP = test_MRR = test_P1 = test_P5 = 0
        start_time = 0
        max_epoch = args.max_epoch
        for epoch in xrange(max_epoch):
            unchanged += 1
            if unchanged > 20: break

            start_time = time.time()

            train = myio.read_annotations(args.train)
            train_batches = myio.create_batches(ids_corpus, train, batch_size,
                                    padding_id, pad_left=not args.average, merge=args.merge)
            N =len(train_batches)

            more = True
            param_bak = [ p.get_value(borrow=False) for p in self.params ]

            while more:

                train_loss = 0.0
                train_cost = 0.0
                train_scost = 0.0
                train_p1 = 0.0

                for i in xrange(N):
                    # get current batch
                    idts, triples, pairs = train_batches[i]

                    cur_cost, cur_loss, cur_scost, cur_p1, gnormg = train_func(idts,
                                                                                triples, pairs)
                    train_loss += cur_loss
                    train_cost += cur_cost
                    train_scost += cur_scost
                    train_p1 += cur_p1

                    if i % 10 == 0:
                        say("\r{}/{} {:.3f}".format(i,N,train_p1/(i+1)))

                cur_train_avg_cost = train_cost / N
                more = False
                if last_train_avg_cost is not None:
                    if cur_train_avg_cost > last_train_avg_cost*(1+tolerance):
                        more = True
                        say("\nTrain cost {} --> {}\n".format(
                                last_train_avg_cost, cur_train_avg_cost
                            ))

                if more:
                    lr_val = lr_g.get_value()*0.5
                    if lr_val < 1e-5: return
                    lr_val = np.float64(lr_val).astype(theano.config.floatX)
                    lr_g.set_value(lr_val)
                    lr_e.set_value(lr_val)
                    say("Decrease learning rate to {}\n".format(float(lr_val)))
                    for p, v in zip(self.params, param_bak):
                        p.set_value(v)
                    continue

                last_train_avg_cost = cur_train_avg_cost

                say("\r\n\n")
                say( ( "Epoch {}  cost={:.3f}  loss={:.3f}  scost={:.3f}" \
                    +"  P[1]={:.3f}  |g|={:.3f}\t[{:.3f}m]\n" ).format(
                        epoch,
                        train_cost / N,
                        train_loss / N,
                        train_scost / N,
                        train_p1 / N,
                        float(gnormg),
                        (time.time()-start_time)/60.0
                ))
                say("\tp_norm: {}\n".format(
                        self.get_pnorm_stat(self.encoder.params)
                    ))
                say("\tp_norm: {}\n".format(
                        self.get_pnorm_stat(self.generator.params)
                    ))

                self.dropout.set_value(0.0)

                if dev is not None:
                    full_MAP, full_MRR, full_P1, full_P5 = self.evaluate(dev, eval_func)
                    dev_MAP, dev_MRR, dev_P1, dev_P5, dev_PZ1, dev_PT = self.evaluate_z(dev,
                            dev_raw, ids_corpus, eval_func2)

                if test is not None:
                    test_MAP, test_MRR, test_P1, test_P5, test_PZ1, test_PT = \
                            self.evaluate_z(test, test_raw, ids_corpus, eval_func2)

                if dev_MAP > best_dev:
                    best_dev = dev_MAP
                    unchanged = 0

                say("\n")
                say("  fMAP={:.2f} fMRR={:.2f} fP1={:.2f} fP5={:.2f}\n".format(
                        full_MAP, full_MRR,
                        full_P1, full_P5
                    ))

                say("\n")
                say(("  dMAP={:.2f} dMRR={:.2f} dP1={:.2f} dP5={:.2f}" +
                     " dP[1]={:.3f} d%T={:.3f} best_dev={:.2f}\n").format(
                        dev_MAP, dev_MRR,
                        dev_P1, dev_P5,
                        dev_PZ1, dev_PT, best_dev
                    ))

                result_table.add_row(
                        [ epoch ] +
                        [ "%.2f" % x for x in [ dev_MAP, dev_MRR, dev_P1, dev_P5 ] +
                                    [ test_MAP, test_MRR, test_P1, test_P5 ] ]
                    )

                if unchanged == 0:
                    say("\n")
                    say(("  tMAP={:.2f} tMRR={:.2f} tP1={:.2f} tP5={:.2f}" +
                        " tP[1]={:.3f} t%T={:.3f}\n").format(
                        test_MAP, test_MRR,
                        test_P1, test_P5,
                        test_PZ1, test_PT
                    ))
                    if args.dump_rationale:
                        self.evaluate_z(dev+test, dev_raw+test_raw, ids_corpus,
                                eval_func2, args.dump_rationale)

                    #if args.save_model:
                    #    self.save_model(args.save_model)

                dropout_p = np.float64(args.dropout).astype(
                            theano.config.floatX)
                self.dropout.set_value(dropout_p)

                say("\n")
                say("{}".format(result_table))
                say("\n")

            if train_p1/N <= 1e-4 or train_p1/N+1e-4 >= 1.0:
                break


    def get_pnorm_stat(self, lst=None):
        lst_norms = [ ]
        if lst is None:
            lst = self.encoder.params + self.generator.params
        for p in lst:
            vals = p.get_value(borrow=True)
            l2 = np.linalg.norm(vals)
            lst_norms.append("{:.3f}".format(l2))
        return lst_norms

    def evaluate(self, data, eval_func):
        res = [ ]
        for idts, labels in data:
            scores = eval_func(idts)
            #print scores.shape, len(labels)
            #print labels
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

    def evaluate_z(self, data, data_raw, ids_corpus,
                            zeval_func, dump_path=None):
        args = self.args
        padding_id = self.padding_id
        tot_p1 = 0.0
        portion_title = 0.0
        tot_selected = 0.0
        res = [ ]
        output_data = [ ]
        for i in xrange(len(data)):
            idts, labels = data[i]
            pid, qids, _ = data_raw[i]
            scores, p1, z = zeval_func(idts)
            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_labels = labels[ranks]
            res.append(ranked_labels)
            tot_p1 += p1

            for wids_i, z_i, question_id in zip(idts.T, z.T, [pid]+qids):
                z2_i = [ zv for wid, zv in zip(wids_i, z_i) if wid != padding_id ]
                title, body = ids_corpus[question_id]
                #portion_title += sum(z2_i[:len(title)])
                if args.merge == 1 or question_id % 2 == 0:
                    portion_title += sum(z2_i[:len(title)])
                else:
                    portion_title += sum(z2_i[-len(title):])
                tot_selected += sum(z2_i)

            if dump_path is not None:
                output_data.append((
                        "Query: ",
                        idts[:,0],
                        z[:,0],
                        pid
                    ))
                for id in ranks[:3]:
                    output_data.append((
                            "Retrieved: {}  label={}".format(scores[id], labels[id]),
                            idts[:,id+1],
                            z[:,id+1],
                            qids[id]
                        ))
        if dump_path is not None:
            embedding_layer = self.embedding_layer
            padding = "<padding>"
            filter_func = lambda w: w != padding
            with open(dump_path, "w") as fout:
                for heading, wordids, z, question_id in output_data:
                    words = embedding_layer.map_to_words(wordids)
                    fout.write(heading+"\tID: {}\n".format(question_id))
                    fout.write("    " + " ".join(filter(filter_func,words)) + "\n")
                    fout.write("------------\n")
                    fout.write("Rationale:\n")
                    fout.write("    " + " ".join(
                            w if zv==1 else "__" for w,zv in zip(words,z) if w != padding
                        ) + "\n")
                    fout.write("\n\n")

        e = Evaluation(res)
        MAP = e.MAP()*100
        MRR = e.MRR()*100
        P1 = e.Precision(1)*100
        P5 = e.Precision(5)*100
        return MAP, MRR, P1, P5, tot_p1/len(data), portion_title/(tot_selected+1e-8)


def main(args):
    raw_corpus = myio.read_corpus(args.corpus)
    embedding_layer = myio.create_embedding_layer(
                raw_corpus,
                n_d = args.hidden_dim,
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
        dev_raw = myio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus, dev_raw, padding_id,
                    pad_left=not args.average, merge=args.merge)
    if args.test:
        test_raw = myio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus, test_raw, padding_id,
                    pad_left=not args.average, merge=args.merge)

    if args.train:
        start_time = time.time()
        train = myio.read_annotations(args.train)
        train_batches = myio.create_batches(ids_corpus, train, args.batch_size,
                                padding_id, pad_left = not args.average, merge=args.merge)
        say("{} to create batches\n".format(time.time()-start_time))
        say("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel()) for x in train_batches),
                sum(len(x[1].ravel()) for x in train_batches)
            ))
        train_batches = None

        model = Model(args, embedding_layer,
                      weights=weights if args.reweight else None)
        model.ready()

        # set parameters using pre-trained network
        if args.load_pretrain:
            model.encoder.load_pretrained_parameters(args)

        model.train(
                ids_corpus,
                train,
                (dev, dev_raw) if args.dev else None,
                (test, test_raw) if args.test else None
            )

if __name__ == "__main__":
    args = options.load_arguments()
    main(args)
