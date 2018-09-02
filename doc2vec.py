import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec

train=pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')
test_id = test["id"].copy()
column="article"

model=Doc2Vec(train[column],vector_size=5,window=2,min_count=1,workers=4)
