# 提取数据集中 word 特征列，除去异常样本并另存
import pandas as pd
from time import time
#s:stop w:word l:list artc:article
maxlen=5000
column="word_seg"
train = pd.read_csv('../data/train_set.csv').drop('article',axis=1)
test = pd.read_csv('../data/test_set.csv').drop('article',axis=1)
count=pd.read_csv("../data/words_count.csv",index_col="No")
keepwords=count[(count>100)&(count<50000)].dropna().index.values.astype(str)
#drop stop words & cut long article
print("lets start")

def DropCut(artc):
    wl=artc.split()
    wl=[w for w in wl if w in keepwords]
    return " ".join(wl[:maxlen])

t1=time()

train[column]=[DropCut(artc) for artc in train[column]]
test[column]=[DropCut(artc) for artc in test[column]]

t2=time()
print("OutlierFilter time use:",t2-t1)

train.to_csv('../data/train_prod.csv',index=None)
test.to_csv('../data/test_prod.csv',index=None)

print("train isnull:",train[column].isnull().sum())
print("test isnull:",test[column].isnull().sum())