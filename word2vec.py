import time, pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import get_tmpfile

t1=time.time()

train = pd.read_csv('../data/train_w5k.csv')
test = pd.read_csv('../data/test_w5k.csv')
column="word_seg"

trn_doc=[sentence.split() for sentence in train[column]]
test_doc=[sentence.split() for sentence in test[column]]
doc=trn_doc+test_doc

model= Word2Vec(workers=4)
model.build_vocab(doc)
model.train(doc,total_examples = model.corpus_count,epochs = model.iter)
model.save('./word2vec.model')
model.wv.save(get_tmpfile("wordvectors.kv"))

t2=time.time()
print("time use:",t2-t1)