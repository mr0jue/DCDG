import pandas as pd
train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')

train_clas=(train["class"]-1).astype(int)
train_clas.to_csv('../data/train_clas.csv',header="class",index=None)
test["id"].to_csv('../data/test_id.csv',header="id",index=None)
print("len(train_clas):",len(train_clas))