
import json

import theano

import myio
from myio import say
from main import Model
from utils import load_embedding_iterator

class QRAPI:

    def __init__(self, model_path, corpus_path, emb_path):
        raw_corpus = myio.read_corpus(corpus_path)
        embedding_layer = myio.create_embedding_layer(
                    raw_corpus,
                    n_d = 10,
                    cut_off = 1,
                    embs = load_embedding_iterator(emb_path)
                )
        weights = myio.create_idf_weights(corpus_path, embedding_layer)
        say("vocab size={}, corpus size={}\n".format(
                embedding_layer.n_V,
                len(raw_corpus)
            ))

        model = Model(args=None, embedding_layer=embedding_layer,
                    weights=weights)

        model_data = model.load_model(model_path)
        model.set_model(model_data)
        model.dropout.set_value(0.0)
        say("model initialized\n")

        score_func = theano.function(
                inputs = [ model.idts, model.idbs ],
                outputs = model.scores,
                on_unused_input='ignore'
            )
        self.model = model
        self.score_func = score_func
        say("scoring function compiled\n")


    def rank(self, query):
        model = self.model
        emb = model.embedding_layer
        args = model.args
        padding_id = model.padding_id
        score_func = self.score_func

        if isinstance(query, str) or isinstance(query, unicode):
            query = json.loads(query)

        p = query["query"].strip().split()
        lst_questions = [ emb.map_to_ids(p, filter_oov=True) ]
        for q in query["candidates"]:
            q = q.strip().split()
            lst_questions.append(
                    emb.map_to_ids(q, filter_oov=True)
                )
        batch, _ = myio.create_one_batch(lst_questions,
                    lst_questions,
                    padding_id,
                    not args.average
                )

        scores = score_func(batch, batch)
        scores = [ x for x in scores ]
        assert len(scores) == len(batch)-1
        if ("BM25" in query) and ("ratio" in query):
            BM25 = query["BM25"]
            ratio = query["ratio"]
            assert len(BM25) == len(scores)
            assert ratio >= 0 and ratio <= 1.0
            scores = [ x*(1-ratio)+y*ratio for x,y in zip(scores, BM25) ]

        ranks = sorted(range(len(scores)), key=lambda i: -scores[i])
        return { "ranks": ranks, "scores": scores }



