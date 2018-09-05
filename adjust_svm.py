import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

print("lets start")

data = pd.read_csv('../data/train_w5k.csv')
X,y=data["word_seg"],(data["class"]-1).astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

t1=time()
vec = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.85,sublinear_tf=True)
train_doc = vec.fit_transform(X_train)
test_doc = vec.transform(X_test)
t2=time()
print("tfidf time use:",t2-t1)
   
def test_svm(ker):
    print("测试开始：SVMkernel=%s"%ker)
    t3=time()
    clf = SVC(kernel=ker,random_state=42)
    clf.fit(train_doc,y_train)
    scores = clf.score(test_doc,y_test)
    t4=time()
    print("svm time use:",t4-t3)
    print("final scores:",scores)

for ker in ['linear','poly','rbf','sigmoid','precomputed']:
   test_svm(ker)

print("all done")