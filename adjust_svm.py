import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

print("lets start")

data = pd.read_csv('../data/train_w5k.csv')
X,y=data["word_seg"],(data["class"]-1).astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

def test_svm(ker):
    print("测试开始：SVMkernel=%s"%ker)
    t1=time()
    vec = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.85,sublinear_tf=True)
    train_doc = vec.fit_transform(X_train)
    test_doc = vec.transform(X_test)
    t2=time()
    print("tfidf time use:",t2-t1)
    clf = SVC(kernel=ker,random_state=42)
    clf.fit(train_doc,y_train)
    scores = clf.score(test_doc,y_test)
    t3=time()
    print("svm time use:",t3-t2)
    print("final scores:",scores)

for ker in ['linear','poly','rbf','sigmoid','precomputed']:
   test_svm(ker)

print("all done")