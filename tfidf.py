import time, pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

t1=time.time()

train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')
test_id = test["id"].copy()
column="word_seg"

n = train.shape
# 将原始文档集合转换为TF-IDF特征矩阵
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
# 这一步的变换是比较耗时的。
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])
