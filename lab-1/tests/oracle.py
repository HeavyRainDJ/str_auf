"""Independent exact-count oracle. Does not replace or patch model submissions."""
from collections import Counter
import re, math
import numpy as np
from scipy.sparse import csr_matrix

WORD = re.compile(r'[A-Za-zА-Яа-яЁё]+')
def tokens(text):
    return WORD.findall(text.lower())

def sparse(counters, features):
    index = {s:i for i,s in enumerate(features)}
    rr, cc, vv = [], [], []
    for row, counter in enumerate(counters):
        for feat, value in counter.items():
            rr.append(row); cc.append(index[feat]); vv.append(value)
    return csr_matrix((vv, (rr, cc)), shape=(len(counters), len(features)), dtype=float)

def expected(texts, nlp):
    names = sorted(texts)
    seqs = [tokens(texts[p]) for p in names]
    words = sorted(set(w for seq in seqs for w in seq))
    char = []
    for w in words:
        s = '^' + w + '$'
        char.append(Counter(s[i:i+n] for n in (2,3) for i in range(len(s)-n+1)))
    feats_char = sorted(set(f for c in char for f in c))
    counts = []
    for seq in seqs:
        counts.append(Counter(seq + [' '.join(pair) for pair in zip(seq, seq[1:])]))
    features = sorted(set(f for c in counts for f in c))
    df = Counter(f for c in counts for f in c)
    weights = []
    for c in counts:
        weight = {f: v * (math.log((1+len(names))/(1+df[f]))+1) for f,v in c.items()}
        norm = math.sqrt(sum(v*v for v in weight.values()))
        weights.append({f:v/norm for f,v in weight.items()} if norm else {})
    wc = {w: Counter() for w in words}; dc = []
    for name in names:
        counter = Counter()
        for dep in nlp(texts[name]):
            head = dep.head
            if dep == head or dep.dep_.lower() in ('root', 'punct'):
                continue
            if not WORD.fullmatch(dep.text) or not WORD.fullmatch(head.text):
                continue
            wc[head.text.lower()][f'OUT|{dep.dep_}|{dep.lemma_.lower()}'] += 1
            wc[dep.text.lower()][f'IN|{dep.dep_}|{head.lemma_.lower()}'] += 1
            counter[f'{head.pos_}|{dep.dep_}|{dep.pos_}'] += 1
        dc.append(counter)
    wf = sorted(set(f for c in wc.values() for f in c)); df = sorted(set(f for c in dc for f in c))
    return {
        'word_char': (sparse(char, feats_char), words, feats_char),
        'doc_tfidf': (sparse(weights, features), names, features),
        'word_syntax': (sparse([wc[w] for w in words], wf), words, wf),
        'doc_syntax': (sparse(dc, df), names, df),
    }
