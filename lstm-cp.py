import sys
from gensim.models import Word2Vec
from LoadData import loadData
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dropout,Dense,Embedding,LSTM,Activation
import pickle
from sklearn.model_selection import train_test_split
from gensim.corpora.dictionary import Dictionary
reload(sys)
sys.setdefaultencoding('utf-8')




# 参数设置
vocab_dim = 100  # 向量维度
maxlen = 150  # 文本保留的最大长度
batch_size = 120
n_epoch = 5
input_length = 150


def text2index(index_dic,sentences):
    """
    把词语转换为数字索引,比如[['中国','安徽','合肥'],['安徽财经大学','今天','天气','很好']]转换为[[1,5,30],[2,3,105,89]]
    """
    new_sentences=[]
    for sen in sentences:
        new_sen=[]
        for word in sen:
            try:
                new_sen.append(index_dic[word])
            except:
                new_sen.append(0)
        new_sentences.append(new_sen)
    return new_sentences


def train_lstm(p_n_symbols, p_embedding_weights, p_X_train, p_y_train, p_X_test, p_y_test):
    """
    :param p_n_symbols: word2vec训练后保留的词语的个数
    :param p_embedding_weights: 词索引与词向量对应矩阵
    :param p_X_train: 训练X
    :param p_y_train: 训练y
    :param p_X_test: 测试X
    :param p_y_test: 测试y
    :return: 
    """
    print u'创建模型...'
    model = Sequential()
    model.add(Embedding(input_dim=p_n_symbols,
                        output_dim=vocab_dim,
                        mask_zero=True,
                        weights=[p_embedding_weights],
                        input_length=input_length,trainable=False))
    model.add(LSTM(units=50))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print u'编译模型...'
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print u"训练..."
    model.fit(p_X_train, p_y_train, batch_size=batch_size, epochs=n_epoch,
              validation_data=(p_X_test, p_y_test),verbose=1)

    print u"评估..."
    score, acc = model.evaluate(p_X_test, p_y_test, batch_size=batch_size)
    print 'Test score:', score
    print 'Test accuracy:', acc


def createModel():
    maxlen=150
    index_dict=pickle.load(open('./w2v_index.pkl','r'))
    vec_dict = pickle.load(open('./w2v_vec.pkl', 'r'))
    n_words=len(index_dict.keys())
    print n_words
    vec_matrix=np.zeros((n_words+1,100))
    for k,i in index_dict.items():#将所有词索引与词向量一一对应
        try:
            vec_matrix[i,:]=vec_dict[k]
        except:
            print k,i
            print vec_dict[k]
            exit(1)
    labels=getLabels()
    sentences=loadData('./sen_cut.txt')
    X_train,X_test,y_train,y_test=train_test_split(sentences,labels,test_size=0.2)
    X_train=text2index(index_dict,X_train)
    X_test = text2index(index_dict, X_test)
    print u"训练集shape： ", np.shape(X_train)
    print u"测试集shape： ", np.shape(X_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)#扩展长度不足的补0
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print u"训练集shape： ", np.shape(X_train)
    print u"测试集shape： ", np.shape(X_test)
    train_lstm(n_words+1, vec_matrix, X_train, y_train, X_test, y_test)


if __name__=="__main__":
    createModel()



y=(train["class"]-1).astype(int)
clf = LogisticRegression(C=4, dual=True)
clf.fit(trn_term_doc, y)
preds=clf.predict_proba(test_term_doc)

#保存概率文件
test_prob=pd.DataFrame(preds)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
test_prob["id"]=list(test_id["id"])
test_prob.to_csv('../data/prob_lr_baseline.csv',index=None)

#生成提交结果
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
print(test_pred.shape)
print(test_id.shape)
test_pred["id"]=list(test_id["id"])
test_pred[["id","class"]].to_csv('../data/sub_lr_baseline.csv',index=None)
t2=time.time()
print("time use:",t2-t1)