import time, pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm

t1=time.time()

train = pd.read_csv('../data/train_set.csv')
test = pd.read_csv('../data/test_set.csv')
test_id = test["id"].copy()

column = "word_seg"

vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

y=(train["class"]-1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc,y)
preds = lin_clf.predict(test_term_doc)

fid=open('../data/tfidf+svm-submit.csv','w')
i=0
fid.write("id,class"+"\n")
for item in preds:
    fid.write(str(i)+","+str(item+1)+"\n")
    i=i+1
fid.close()

t2=time.time()
print("time use:",t2-t1)