import sys
import gzip
import random
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import theano

from nn import EmbeddingLayer

def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()

def read_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                print id
                empty_cnt += 1
                continue
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = (title, body)
    say("{} empty titles ignored.\n".format(empty_cnt))
    return raw_corpus

def create_embedding_layer(raw_corpus, n_d, embs=None, \
        cut_off=2, unk="<unk>", padding="<padding>", fix_init_embs=True):

    cnt = Counter(w for id, pair in raw_corpus.iteritems() \
                        for x in pair for w in x)
    cnt[unk] = cut_off + 1
    cnt[padding] = cut_off + 1
    cnt["<s>"] = cut_off + 1
    cnt["</s>"] = cut_off + 1
    embedding_layer = EmbeddingLayer(
            n_d = n_d,
            #vocab = (w for w,c in cnt.iteritems() if c > cut_off),
            vocab = [ unk, padding, "<s>", "</s>" ],
            embs = embs,
            fix_init_embs = fix_init_embs
        )
    return embedding_layer

def create_idf_weights(corpus_path, embedding_layer):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,1), binary=False)

    lst = [ ]
    fopen = gzip.open if corpus_path.endswith(".gz") else open
    with fopen(corpus_path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            lst.append(title)
            lst.append(body)
    vectorizer.fit_transform(lst)

    idfs = vectorizer.idf_
    avg_idf = sum(idfs)/(len(idfs)+0.0)/4.0
    weights = np.array([ avg_idf for i in xrange(embedding_layer.n_V) ],
                    dtype = theano.config.floatX)
    vocab_map = embedding_layer.vocab_map
    for word, idf_value in zip(vectorizer.get_feature_names(), idfs):
        id = vocab_map.get(word, -1)
        if id != -1:
            weights[id] = idf_value
    return theano.shared(weights, name="word_weights")

def map_corpus(raw_corpus, embedding_layer, max_len=100):
    ids_corpus = { }
    for id, pair in raw_corpus.iteritems():
        ids_corpus[id] = (embedding_layer.map_to_ids(pair[0], filter_oov=True),
                          embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len])
    return ids_corpus

def read_annotations(path, K_neg=0, prune_pos_cnt=10):
    lst = [ ]
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            neg = neg[:K_neg]
            s = set()
            qids = [ ]
            qlabels = [ ]
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            lst.append((pid, qids, qlabels))
    return lst

def create_batches(ids_corpus, data, batch_size, padding_id,
                        bos_id, eos_id, auto_encode=True, perm=None):
    if perm is None:
        perm = range(len(data))
        random.shuffle(perm)
    pair_set = set()
    for pid, qids, qlabels in data:
        for qid in qids:
            pair_set.add((pid, qid))
            pair_set.add((qid, pid))
    pair_lst = list(pair_set)
    if auto_encode:
        for pid in ids_corpus:
            pair_lst.append((pid, pid))
    random.shuffle(pair_lst)
    random.shuffle(pair_lst)
    cnt = 0
    titles1 = [ ]
    titles2 = [ ]
    bodies1 = [ ]
    #bodies2 = [ ]
    batches = [ ]
    N = len(pair_lst)
    for i in xrange(N):
        pid, qid = pair_lst[i]
        if pid not in ids_corpus: continue
        if qid not in ids_corpus: continue
        t, b = ids_corpus[pid]
        titles1.append(t)
        bodies1.append(b)
        t, b = ids_corpus[qid]
        titles2.append(np.hstack([bos_id, t, eos_id]).astype('int32'))
        #bodies2.append(b)
        cnt += 1
        if cnt == batch_size or i == N-1:
            #batches.append((titles1, bodies1, titles2, bodies2))
            batches.append((titles1, bodies1, titles2))
            titles1 = [ ]
            titles2 = [ ]
            bodies1 = [ ]
            #bodies2 = [ ]
            cnt = 0
    return batches

def create_eval_batches(ids_corpus, data, padding_id):
    lst = [ ]
    for pid, qids, qlabels in data:
        t1, b1 = ids_corpus[pid]
        titles = [ t1 ]
        bodies = [ b1 ]
        for qid in qids:
            t2, b2 = ids_corpus[qid]
            titles.append(t2)
            bodies.append(b2)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
    return lst

def create_one_batch(source, target, padding_id):
    max_source_len = max(1, max(len(x) for x in source))
    max_target_len = max(1, max(len(x) for x in target))
    source = np.column_stack([ np.pad(x,(max_source_len-len(x),0),'constant',
                            constant_values=padding_id) for x in source])
    target = np.column_stack([ np.pad(x,(0,max_target_len-len(x)),'constant',
                            constant_values=padding_id) for x in target])
    return source, target
