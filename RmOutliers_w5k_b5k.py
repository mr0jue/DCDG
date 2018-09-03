# 提取数据集中 word 特征列，除去异常样本并另存
import pandas as pd
from time import time
#s:stop w:word l:list artc:article
maxlen=5000
column="word_seg"
train = pd.read_csv('../data/train_set.csv').drop('article',axis=1)
test = pd.read_csv('../data/test_set.csv').drop('article',axis=1)
stopwords=pd.read_csv("../data/stopwords_b5k.csv").values
stopwords=[sw[0].astype(str) for sw in stopwords]
#drop stop words & cut long article
def DropCut(artc):
    wl=artc.split()
    wl=[w for w in wl if w not in stopwords]
    return " ".join(wl[:maxlen])

t1=time()

train[column]=[DropCut(artc) for artc in train[column]]
test[column]=[DropCut(artc) for artc in test[column]]

t2=time()
print("OutlierFiltering time use:",t2-t1)

train.to_csv('../data/train_w5k_b5k.csv',index=None)
test.to_csv('../data/test_w5k_b5k.csv',index=None)

print("train.shape:",train.shape)