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
            id = int(id)
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
        cut_off=5, unk="<unk>", padding="<padding>", fix_init_embs=True):

    cnt = Counter(w for id, pair in raw_corpus.iteritems() \
                        for x in pair for w in x)
    cnt[unk] = cut_off + 1
    cnt[padding] = cut_off + 1
    embedding_layer = EmbeddingLayer(
            n_d = n_d,
            vocab = (w for w,c in cnt.iteritems() if c > cut_off),
            #vocab = [ unk, padding ],
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
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=False),
                          embedding_layer.map_to_ids(pair[1], filter_oov=False)[:max_len])
        ids_corpus[id] = item
    return ids_corpus

def read_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = [ ]
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pid = int(pid)
            pos = map(int, pos.split())
            neg = map(int, neg.split())
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1): continue
            if K_neg != -1:
                random.shuffle(neg)
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

def create_batches(ids_corpus, data, batch_size, padding_id, \
                        perm=None, pad_left=True, merge=0):
    if perm is None:
        perm = range(len(data))
        random.shuffle(perm)

    N = len(data)
    cnt = 0
    pid2id = {}
    titles = [ ]
    bodies = [ ]
    triples = [ ]
    batches = [ ]
    pairs = [ ]
    for u in xrange(N):
        i = perm[u]
        pid, qids, qlabels = data[i]
        if pid not in ids_corpus: continue
        cnt += 1
        pos = 0
        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus: continue
                id_in_batch = len(titles)
                pid2id[id] = id_in_batch
                t, b = ids_corpus[id]
                if not merge:
                    titles.append(t)
                    bodies.append(b)
                else:
                    item = np.append(t,b) if random.random()>0.5 else np.append(b,t)
                    titles.append(item)
                if id == pid or qlabels[pos-1] == 0:
                    pairs.append([id_in_batch, id_in_batch])
                else:
                    pairs.append([id_in_batch, pid2id[pid]])
            pos += 1
        pid = pid2id[pid]
        pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ]
        neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ]
        triples += [ [pid,x]+neg for x in pos ]
        #triples += [ [pid, x, y] for x in pos for y in neg ]

        if cnt == batch_size or u == N-1:
            triples = create_hinge_batch(triples)
            #triples = np.asarray(triples, dtype="int32")
            pairs = np.asarray(pairs, dtype="int32")
            if not merge:
                titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
                batches.append((titles, bodies, triples, pairs))
            else:
                texts = create_one_batch(titles, None, padding_id, pad_left)
                batches.append((texts, triples, pairs))
            titles = [ ]
            bodies = [ ]
            triples = [ ]
            pairs = [ ]
            pid2id = {}
            cnt = 0
    return batches

def create_eval_batches(ids_corpus, data, padding_id, pad_left, merge=0):
    lst = [ ]
    for pid, qids, qlabels in data:
        titles = [ ]
        bodies = [ ]
        for id in [pid]+qids:
            t, b = ids_corpus[id]
            if not merge:
                titles.append(t)
                bodies.append(b)
            else:
                item = np.append(t,b) if merge==1 or id%2==0 else np.append(b,t)
                #item = np.append(t,b)
                titles.append(item)
        if not merge:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
        else:
            texts = create_one_batch(titles, None, padding_id, pad_left)
            lst.append((texts, np.array(qlabels, dtype="int32")))
    return lst

def create_one_batch(titles, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles))
    if pad_left:
        titles = np.column_stack([ np.pad(x,(max_title_len-len(x),0),'constant',
                                constant_values=padding_id) for x in titles])
    else:
        titles = np.column_stack([ np.pad(x,(0,max_title_len-len(x)),'constant',
                                constant_values=padding_id) for x in titles])

    if bodies is None:
        return titles

    max_body_len = max(1, max(len(x) for x in bodies))
    if pad_left:
        bodies = np.column_stack([ np.pad(x,(max_body_len-len(x),0),'constant',
                                constant_values=padding_id) for x in bodies])
    else:
        bodies = np.column_stack([ np.pad(x,(0,max_body_len-len(x)),'constant',
                                constant_values=padding_id) for x in bodies])
    return titles, bodies

def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                        for x in triples ]).astype('int32')
    return triples
