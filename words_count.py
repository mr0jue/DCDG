import pandas as pd
from time import time

column="word_seg"
train = pd.read_csv('../data/train_w5k.csv')
test = pd.read_csv('../data/test_w5k.csv')

t1=time()
wll=[wdl.split() for wdl in train[column]]
wwh=[]
for wl in wll:
    wwh.extend(wl)
count=pd.Series(wwh).value_counts()
count.index.name="No"
count.name="counts"
count.to_csv("../data/words_count.csv",header=True)
print("words count done.")
t2=time()
print("time use:",t2-t1)