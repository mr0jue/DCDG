import time, pandas as pd
from gensim.models import KeyedVectors# Word2Vec
from gensim.test.utils import get_tmpfile

t1=time.time()

train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')
test_id = test["id"].copy()
column="word_seg"

wv= KeyedVectors.load(get_tmpfile("../data/wordvectors.kv"),mmap="r")




t2=time.time()
print("time use:",t2-t1)