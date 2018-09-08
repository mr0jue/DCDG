# 提取数据集中 word 特征列，除去异常样本并另存
import pandas as pd
from time import time
#s:stop w:word l:list artc:article
maxlen=5000
column="word_seg"
train = pd.read_csv('../data/train_set.csv',chunksize=10000)
test = pd.read_csv('../data/test_set.csv',chunksize=10000)
count=pd.read_csv("../data/words_count.csv",index_col="No")
keepwords=count[(count>100)&(count<50000)].dropna().index.values.astype(str)
#drop stop words & cut long article
print("lets start")

def DropCut(artc):
    wl=artc.split()
    wl=[w for w in wl if w in keepwords]
    wl=" ".join(wl[:maxlen])
    return wl

def Pro(dats):
	chunks = []
	for dat in dats:
		dat[column]=[DropCut(artc) for artc in dat[column]]
		chunks.append(dat)
	dats = pd.concat(chunks, ignore_index=True)
	return dats

t1=time()
print("pro train")
train=Pro(train)
print("pro test")
test=Pro(test)
t2=time()
print("OutlierFilter time use:",t2-t1)

train.drop('article',axis=1).to_csv('../data/train_prod.csv',index=None)
test.drop('article',axis=1).to_csv('../data/test_prod.csv',index=None)

print("train isnull:",train[column].isnull().sum())
print("test isnull:",test[column].isnull().sum())