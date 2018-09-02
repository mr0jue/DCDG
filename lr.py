import pandas as pd, numpy as np
from scipy import sparse
from time import time
from sklearn.linear_model import LogisticRegression

test_id=pd.read_csv('../data/test_id.csv')["id"]
y=pd.read_csv('../data/train_clas.csv')["class"]
trn_term_doc=sparse.load_npz('../data/trn_term_doc_w10k.npz')
test_term_doc=sparse.load_npz('../data/test_term_doc_w10k.npz')

t1=time()

clf = LogisticRegression(C=4, dual=True)
clf.fit(trn_term_doc, y)
preds = clf.predict_proba(test_term_doc)

t2=time()
print("lr time use:",t2-t1)

test_prob=pd.DataFrame(preds)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
test_prob["id"]=test_id
test_prob.to_csv('../data/tfidf+lr_prob.csv',index=None)

test_pred=pd.DataFrame(np.argmax(preds,axis=1))
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
test_pred["id"]=test_id
test_pred[["id","class"]].to_csv('../data/tfidf+lr_sub.csv',index=None)

print("probs shape:",test_prob.shape)
print("preds shape:",test_pred.shape)