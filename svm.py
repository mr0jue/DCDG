import pandas as pd
from scipy import sparse
from time import time
from sklearn.svm import LinearSVC

test_id=pd.read_csv('../data/test_id.csv')["id"]
y=pd.read_csv('../data/train_clas.csv')["class"]
trn_term_doc=sparse.load_npz('../data/trn_term_doc_w10k.npz')
test_term_doc=sparse.load_npz('../data/test_term_doc_w10k.npz')

t1=time()

clf = LinearSVC()
clf.fit(trn_term_doc, y)
preds = clf.predict(test_term_doc)

t2=time()
print("svm time use:",t2-t1)

test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
test_pred["id"]=test_id
test_pred[["id","class"]].to_csv('../data/tfidf+svm_sub.csv',index=None)

print("preds shape:",test_pred.shape)