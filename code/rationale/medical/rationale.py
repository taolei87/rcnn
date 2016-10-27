
import os, sys, gzip
import time
import math
import json

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compile.nanguardmode import NanGuardMode

from nn import create_optimization_updates, get_activation_by_name, sigmoid, linear
from nn import EmbeddingLayer, Layer, apply_dropout, default_rng
from utils import say
import myio
import options

class CNN(Layer):
    '''
        CNN implementation. Return feature maps over time. No pooling is used.

        Inputs
        ------

            order       : filter width = 2*order-1
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            order=1, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients

        self.input_shape = (None, 1, n_in, None)
        self.filter_shape = (n_out, 1, n_in, order*2-1)
        self.W = create_shared(random_init(self.filter_shape), name="W")
        self.bias = create_shared(random_init((n_out,)), name="bias")

    def forward_all(self, x):

        # x is len*batch*d
        if self.order>1:
            padding = T.zeros((self.order-1,x.shape[1],x.shape[2]),
                                dtype=theano.config.floatX)
            xs = T.concatenate([ padding, x, padding ], axis=0)
        else:
            xs = x
        xs = xs.dimshuffle((1,'x',2,0))

        # batch*d*1*len
        conv_out = T.nnet.conv2d(
                input = xs,
                filters = self.W,
                image_shape = self.input_shape,
                filter_shape = self.filter_shape,
                border_mode = 'valid'
        )
        conv_out = self.activation(conv_out + self.bias.dimshuffle(('x',0,'x','x')))

        # batch*d*len
        h = conv_out.flatten(3)

        # len*batch*d
        return h.dimshuffle((2,0,1))

    @property
    def params(self):
        return [ self.W, self.bias ]

    @params.setter
    def params(self, param_list):
        assert len(param_list) == 2
        self.W.set_value(param_list[0].get_value())
        self.bias.set_value(param_list[1].get_value())


class Generator(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses

    def ready(self):
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]

        dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

        # len*batch
        x = self.x = T.imatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]
        layer_type = args.layer.lower()
        for i in xrange(1):
            l = CNN(
                    n_in = n_e,
                    n_out = n_d,
                    activation = activation,
                    order = args.order
                )
            layers.append(l)

        # len * batch
        masks = T.cast(T.neq(x, padding_id), "int8").dimshuffle((0,1,'x'))

        # (len*batch)*n_e
        embs = embedding_layer.forward(x.ravel())
        # len*batch*n_e
        embs = embs.reshape((x.shape[0], x.shape[1], n_e))
        embs = apply_dropout(embs, dropout)
        self.word_embs = embs

        # len*bacth*n_d
        h1 = layers[0].forward_all(embs)
        h_final = h1
        size = n_d
        h_final = apply_dropout(h_final, dropout)

        output_layer = self.output_layer = Layer(
                n_in = size,
                n_out = 1,
                activation = sigmoid
            )

        # len*batch*1
        probs = output_layer.forward(h_final)

        # len*batch
        self.MRG_rng = MRG_RandomStreams()
        z_pred_dim3 = self.MRG_rng.binomial(size=probs.shape, p=probs, dtype="int8")
        z_pred = z_pred_dim3.reshape(x.shape)

        # we are computing approximated gradient by sampling z;
        # so should mark sampled z not part of the gradient propagation path
        #
        z_pred = self.z_pred = theano.gradient.disconnected_grad(z_pred)
        print "z_pred", z_pred.ndim

        #logpz = - T.nnet.binary_crossentropy(probs, z_pred_dim3) * masks
        logpz = - T.nnet.binary_crossentropy(probs, z_pred_dim3)
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


class Encoder(object):

    def __init__(self, args, embedding_layer, nclasses, generator):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        self.generator = generator

    def ready(self):
        generator = self.generator
        embedding_layer = self.embedding_layer
        args = self.args
        padding_id = embedding_layer.vocab_map["<padding>"]
        unk_id = embedding_layer.vocab_map["<unk>"]
        unk_vec = embedding_layer.embeddings[unk_id]

        dropout = generator.dropout

        # len*batch
        x = generator.x
        z = generator.z_pred
        z = z.dimshuffle((0,1,"x"))

        # batch*nclasses
        y = self.y = T.fmatrix()

        n_d = args.hidden_dimension
        n_e = embedding_layer.n_d
        activation = get_activation_by_name(args.activation)

        layers = self.layers = [ ]
        depth = args.depth
        layer_type = args.layer.lower()
        for i in xrange(depth):
            l = CNN(
                    n_in = n_e if i == 0 else n_d,
                    n_out = n_d,
                    activation = activation,
                    order = args.order
                )
            layers.append(l)

        # len * batch * 1
        masks = T.cast(T.neq(x, padding_id).dimshuffle((0,1,"x")) * z, theano.config.floatX)
        # batch * 1
        cnt_non_padding = T.sum(masks, axis=0) + 1e-8

        # len*batch*n_e
        embs = generator.word_embs*z + unk_vec.dimshuffle(('x','x',0))*(1-z)

        pooling = args.pooling
        lst_states = [ ]
        h_prev = embs
        for l in layers:
            # len*batch*n_d
            h_next = l.forward_all(h_prev)
            if pooling:
                # batch * n_d
                masked_sum = T.sum(h_next * masks, axis=0)
                lst_states.append(masked_sum/cnt_non_padding) # mean pooling
            else:
                lst_states.append(T.max(h_next, axis=0))
            h_prev = apply_dropout(h_next, dropout)

        if args.use_all:
            size = depth * n_d
            # batch * size (i.e. n_d*depth)
            h_final = T.concatenate(lst_states, axis=1)
        else:
            size = n_d
            h_final = lst_states[-1]
        h_final = apply_dropout(h_final, dropout)

        output_layer = self.output_layer = Layer(
                n_in = size,
                n_out = self.nclasses,
                activation = sigmoid
            )

        # batch * nclasses
        p_y_given_x = self.p_y_given_x = output_layer.forward(h_final)
        preds = self.preds = p_y_given_x > 0.5
        print preds, preds.dtype
        print self.nclasses

        # batch
        loss_mat = T.nnet.binary_crossentropy(p_y_given_x, y)

        if args.aspect < 0:
            loss_vec = T.mean(loss_mat, axis=1)
        else:
            assert args.aspect < self.nclasses
            loss_vec = loss_mat[:,args.aspect]
        self.loss_vec = loss_vec

        self.true_pos = T.sum(preds*y)
        self.tot_pos = T.sum(preds)
        self.tot_true = T.sum(y)

        zsum = generator.zsum
        zdiff = generator.zdiff
        logpz = generator.logpz

        coherent_factor = args.sparsity * args.coherent
        loss = self.loss = T.mean(loss_vec)
        sparsity_cost = self.sparsity_cost = T.mean(zsum) * args.sparsity + \
                                             T.mean(zdiff) * coherent_factor
        cost_vec = loss_vec + zsum * args.sparsity + zdiff * coherent_factor
        cost_logpz = T.mean(cost_vec * T.sum(logpz, axis=0))
        self.obj = T.mean(cost_vec)

        params = self.params = [ ]
        for l in layers + [ output_layer ]:
            for p in l.params:
                params.append(p)
        if not args.fix_emb:
            params += embedding_layer.params
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

        self.cost_g = cost_logpz + generator.l2_cost
        self.cost_e = loss + l2_cost

class Model(object):

    def __init__(self, args, embedding_layer, nclasses):
        self.args = args
        self.embedding_layer = embedding_layer
        self.nclasses = nclasses
        self.generator = Generator(args, embedding_layer, nclasses)
        self.encoder = Encoder(args, embedding_layer, nclasses, self.generator)


    def ready(self):
        self.generator.ready()
        self.encoder.ready()
        self.dropout = self.generator.dropout
        self.x = self.generator.x
        self.y = self.encoder.y
        self.z = self.generator.z_pred
        self.params = self.encoder.params + self.generator.params
        self.true_pos = self.encoder.true_pos
        self.tot_pos = self.encoder.tot_pos
        self.tot_true = self.encoder.tot_true

    def train(self, train, dev, test):
        args = self.args
        dropout = self.dropout
        padding_id = self.embedding_layer.vocab_map["<padding>"]

        if dev is not None:
            dev_batches_x, dev_batches_y = myio.create_batches(
                            dev[0], dev[1], args.batch, padding_id
                        )
        if test is not None:
            test_batches_x, test_batches_y = myio.create_batches(
                            test[0], test[1], args.batch, padding_id
                        )

        start_time = time.time()
        train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )
        say("{:.2f}s to create training batches\n\n".format(
                time.time()-start_time
            ))

        updates_e, lr_e, gnorm_e = create_optimization_updates(
                               cost = self.encoder.cost_e,
                               params = self.encoder.params,
                               method = args.learning,
                               lr = args.learning_rate
                        )[:3]


        updates_g, lr_g, gnorm_g = create_optimization_updates(
                               cost = self.encoder.cost_g,
                               params = self.generator.params,
                               method = args.learning,
                               lr = args.learning_rate
                        )[:3]

        sample_generator = theano.function(
                inputs = [ self.x ],
                outputs = self.z
            )

        get_loss_and_pred = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.encoder.loss_vec, self.encoder.preds, self.z ]
            )

        train_generator = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.encoder.obj, self.encoder.loss, \
                                self.encoder.sparsity_cost, self.z, gnorm_e, gnorm_g ],
                updates = updates_e.items() + updates_g.items(),
            )

        eval_func = theano.function(
                inputs = [ self.x, self.y ],
                outputs = [ self.z, self.encoder.obj, self.true_pos, self.tot_pos, self.tot_true ]
            )

        eval_period = args.eval_period
        unchanged = 0
        best_dev = 1e+2
        best_dev_e = 1e+2
        last_train_avg_cost = None
        last_dev_avg_cost = None
        tolerance = 0.10 + 1e-3
        dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

        for epoch in xrange(args.max_epochs):
            unchanged += 1
            if unchanged > 50: return

            train_batches_x, train_batches_y = myio.create_batches(
                            train[0], train[1], args.batch, padding_id
                        )

            more = True
            if args.decay_lr:
                param_bak = [ p.get_value(borrow=False) for p in self.params ]

            while more:
                processed = 0
                train_cost = 0.0
                train_loss = 0.0
                train_sparsity_cost = 0.0
                p1 = 0.0
                start_time = time.time()

                N = len(train_batches_x)
                for i in xrange(N):
                    if (i+1) % 100 == 0:
                        say("\r{}/{} {:.2f}       ".format(i+1,N,p1/(i+1)))

                    bx, by = train_batches_x[i], train_batches_y[i]
                    mask = bx != padding_id

                    cost, loss, sparsity_cost, bz, gl2_e, gl2_g = train_generator(bx, by)

                    k = len(by)
                    processed += k
                    train_cost += cost
                    train_loss += loss
                    train_sparsity_cost += sparsity_cost
                    p1 += np.sum(bz*mask) / (np.sum(mask)+1e-8)

                cur_train_avg_cost = train_cost / N

                if dev:
                    self.dropout.set_value(0.0)
                    dev_obj, dev_prec, dev_recall, dev_f1, dev_p1 = self.evaluate_data(
                            dev_batches_x, dev_batches_y, eval_func)
                    self.dropout.set_value(dropout_prob)
                    cur_dev_avg_cost = dev_obj

                more = False
                if args.decay_lr and last_train_avg_cost is not None:
                    if cur_train_avg_cost > last_train_avg_cost*(1+tolerance):
                        more = True
                        say("\nTrain cost {} --> {}\n".format(
                                last_train_avg_cost, cur_train_avg_cost
                            ))
                    if dev and cur_dev_avg_cost > last_dev_avg_cost*(1+tolerance):
                        more = True
                        say("\nDev cost {} --> {}\n".format(
                                last_dev_avg_cost, cur_dev_avg_cost
                            ))

                if more:
                    lr_val = lr_g.get_value()*0.5
                    lr_val = np.float64(lr_val).astype(theano.config.floatX)
                    lr_g.set_value(lr_val)
                    lr_e.set_value(lr_val)
                    say("Decrease learning rate to {}\n".format(float(lr_val)))
                    for p, v in zip(self.params, param_bak):
                        p.set_value(v)
                    continue

                last_train_avg_cost = cur_train_avg_cost
                if dev: last_dev_avg_cost = cur_dev_avg_cost

                say("\n")
                say(("Generator Epoch {:.2f}  costg={:.4f}  scost={:.4f}  lossg={:.4f}  " +
                    "p[1]={:.3f}  |g|={:.4f} {:.4f}\t[{:.2f}m / {:.2f}m]\n").format(
                        epoch+(i+1.0)/N,
                        train_cost / N,
                        train_sparsity_cost / N,
                        train_loss / N,
                        p1 / N,
                        float(gl2_e),
                        float(gl2_g),
                        (time.time()-start_time)/60.0,
                        (time.time()-start_time)/60.0/(i+1)*N
                    ))
                say("\t"+str([ "{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                for x in self.encoder.params ])+"\n")
                say("\t"+str([ "{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
                                for x in self.generator.params ])+"\n")

                if dev:
                    if dev_obj < best_dev:
                        best_dev = dev_obj
                        unchanged = 0
                        if args.dump and test:
                            self.dump_rationales(args.dump, test_batches_x, test_batches_y,
                                        get_loss_and_pred, sample_generator)

                    say(("\tdevg={:.4f}  f1g={:.4f}  preg={:.4f}  recg={:.4f}" +
                                "  p[1]g={:.3f}  best_dev={:.4f}\n").format(
                        dev_obj,
                        dev_f1,
                        dev_prec,
                        dev_recall,
                        dev_p1,
                        best_dev
                    ))

                    if test is not None:
                        self.dropout.set_value(0.0)
                        test_obj, test_prec, test_recall, test_f1, test_p1 = self.evaluate_data(
                            test_batches_x, test_batches_y, eval_func)
                        self.dropout.set_value(dropout_prob)
                        say(("\ttestt={:.4f}  f1t={:.4f}  pret={:.4f}  rect={:.4f}" +
                                    "  p[1]t={:.3f}\n").format(
                            test_obj,
                            test_f1,
                            test_prec,
                            test_recall,
                            test_p1
                        ))

    def evaluate_data(self, batches_x, batches_y, eval_func):
        tot_loss = 0.0
        true_pos, cnt_pos, cnt_true = 0.0, 0.0, 0.0
        p1 = 0.0
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        for bx, by in zip(batches_x, batches_y):
            bz, l, tp, cp, ct = eval_func(bx, by)
            assert(tp <= cp and tp <= ct)
            tot_loss += l
            true_pos += tp
            cnt_pos += cp
            cnt_true += ct
            mask = bx != padding_id
            p1 += np.sum(bz*mask) / (np.sum(mask) + 1e-8)
        prec = true_pos/cnt_pos
        recall = true_pos/cnt_true
        f1 = prec*recall*2/(prec+recall)
        n = len(batches_x)
        return tot_loss/n, prec, recall, f1, p1/n

    def dump_rationales(self, path, batches_x, batches_y, eval_func, sample_func):
        embedding_layer = self.embedding_layer
        padding_id = self.embedding_layer.vocab_map["<padding>"]
        lst = [ ]
        for bx, by in zip(batches_x, batches_y):
            loss_vec_r, preds_r, bz = eval_func(bx, by)
            assert len(loss_vec_r) == bx.shape[1]
            for loss_r, p_r, x,y,z in zip(loss_vec_r, preds_r, bx.T, by, bz.T):
                loss_r = float(loss_r)
                p_r, x, y, z = p_r.tolist(), x.tolist(), y.tolist(), z.tolist()
                w = embedding_layer.map_to_words(x)
                w = [ u if u!="<padding>" else "_" for u in w ]
                r = [ u if v == 1 else "_" for u,v in zip(w,z) ]
                diff = max(y)-min(y)
                lst.append((diff, loss_r, r, w, x, y, z, p_r))

        #lst = sorted(lst, key=lambda x: (len(x[3]), x[2]))
        with open(path,"w") as fout:
            for diff, loss_r, r, w, x, y, z, p_r in lst:
                fout.write( json.dumps( { "rationale": " ".join(r),
                                          "text": " ".join(w),
                                          #"x": x,
                                          #"z": z,
                                          "y": y,
                                          "p": p_r
                                          } ) + "\n\n" )


def main():
    print args
    assert args.embedding, "Pre-trained word embeddings required."

    embedding_layer = myio.create_embedding_layer(
                        args.embedding
                    )

    max_len = args.max_len

    if args.train:
        train_x, train_y = myio.read_annotations(args.train)
        train_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in train_x ]

    if args.dev:
        dev_x, dev_y = myio.read_annotations(args.dev)
        dev_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in dev_x ]

    if args.test:
        test_x, test_y = myio.read_annotations(args.test)
        test_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in test_x ]

    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
                    nclasses = len(train_y[0])
                )
        model.ready()

        #debug_func2 = theano.function(
        #        inputs = [ model.x, model.z ],
        #        outputs = model.generator.logpz
        #    )
        #theano.printing.debugprint(debug_func2)
        #return

        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                (test_x, test_y) if args.test else None
            )

if __name__=="__main__":
    args = options.load_arguments()
    main()
