# 提取数据集中 word 特征列，除去异常样本并另存
import pandas as pd
from time import time
#.drop('article',axis=1)
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

column="word_seg"
maxlen=10000

def Cut(wdl):
	if len(wdl.split())<maxlen:
		return wdl
	else:
		return " ".join(wdl.split()[:maxlen])

t1=time()

train[column]=[Cut(wdl) for wdl in train[column]]
test[column]=[Cut(wdl) for wdl in test[column]]

t2=time()
print("OutlierFiltering time use:",t2-t1)

train.drop('article',axis=1).to_csv('../data/train_w10k.csv',index=None)
test.drop('article',axis=1).to_csv('../data/test_w10k.csv',index=None)

print("train.shape:",train.shape)