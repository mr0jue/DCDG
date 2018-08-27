import pandas as pd
import seaborn as sns  
import matplotlib.plot as plt

train=pd.read_csv("../data/train_set.csv")
print("打印 train_set 的前五行：")
print(train.head())

clas=train["class"].value_counts()
print("类别	该类别样本数")
print(clas)

f,axes = plt.subplots(2,2,figsize=(7,7),sharex=True)
l=[]
for i in [0,1]:
    for j in [1,2]:
        nob=train.iloc[i,j].split(' ')
        l.append(len(nob))
        sns.distplot(pd.Series(map(lambda x:int(x),nob)),ax=axes[i,j-1])
print("id0字数	id0词数	id1字数	id1词数")
print(l)
plt.show()