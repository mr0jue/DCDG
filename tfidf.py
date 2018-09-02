# 对文档执行TF-IDF过程，并保存结果
import pandas as pd
from scipy import sparse
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer

t1=time()

column="word_seg"
train = pd.read_csv('../data/train_w10k.csv')
test = pd.read_csv('../data/test_w10k.csv')

vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

t2=time()
print("tfidf time use:",t2-t1)

sparse.save_npz('../data/trn_term_doc_w10k.npz', trn_term_doc) 
sparse.save_npz('../data/test_term_doc_w10k.npz', test_term_doc) 

print("train.shape:",train.shape)