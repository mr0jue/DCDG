import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
t1=time.time()
train = pd.read_csv('input/train_set.csv')[:100]
test = pd.read_csv('input/test_set.csv')[:10]
test_id = pd.read_csv('input/test_set.csv')[["id"]][:10].copy()

column="word_seg"

# 用来查找数据的维度
n = train.shape
# 将原始文档集合转换为TF-IDF特征矩阵
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
# 这一步的变换是比较耗时的。
# 为何只是对word_seg进行了变换呢？
trn_term_doc = vec.fit_transform(train[column])
# 这个和上面的为何又是不一样呢？
test_term_doc = vec.transform(test[column])


y=(train["class"]-1).astype(int)
# 创建逻辑回归分类器
clf = LogisticRegression(C=4,random_state=7,class_weight='balanced',solver="sag",
                         multi_class="multinomial",n_jobs=-1)
clf.fit(trn_term_doc, y)
# 得到预测的概率
preds=clf.predict_proba(test_term_doc)


#保存概率文件
# 先把概率转换为pandas数据类型
test_prob=pd.DataFrame(preds)
# print(list(test_prob.columns))

# test_prob.columns的结果是(start=0,stop=1,step=1)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
# print(test_prob.columns)
# 输出的结果是；
# Index(['class_prob_1', 'class_prob_2', 'class_prob_3', 'class_prob_4',
#        'class_prob_5', 'class_prob_6', 'class_prob_7', 'class_prob_8',
#        'class_prob_9', 'class_prob_10', 'class_prob_11', 'class_prob_12',
#        'class_prob_13', 'class_prob_14', 'class_prob_15', 'class_prob_16',
#        'class_prob_17', 'class_prob_18'],
#       dtype='object')

test_prob["id"]=list(test_id["id"])
# print(test_prob["id"])
test_prob.to_csv('input/prob_lr_baseline.csv',index=None)
#
# #生成提交结果
preds=np.argmax(preds,axis=1)
# [13  8 11 12  8  2  2 12  2 12]
print(preds)
test_pred=pd.DataFrame(preds)
print(test_pred)  # 0
test_pred.columns=["class"]


test_pred["class"]=(test_pred["class"]+1).astype(int)
# print(test_pred["class"])
# 0    14
# 1     9
# 2    12
# 3    13
# 4     9
# 5     3
# 6     3
# 7    13
# 8     3
# 9    13
# 这里打印的是预测结果的
print(test_pred.shape)  # （10，1）
# 这里打印的是所有的id维度
print(test_id.shape)  # （10，1），这个结果是因为我设置的[:10]所以是（10，1）
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('input/sub_lr_baseline.csv',index=None)
t2=time.time()
print("time use:",t2-t1)
