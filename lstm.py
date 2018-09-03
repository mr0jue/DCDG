import time, numpy as np, pandas as pd
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical

t1=time.time()

path = get_tmpfile("./wordvectors.kv")
wv = KeyedVectors.load(path, mmap='r')

train = pd.read_csv('../data/train_w5k.csv')
test = pd.read_csv('../data/test_w5k.csv')
test_id = test["id"].copy()
y=to_categorical((train["class"]-1).astype(int))
column="word_seg"
print("clashape:",y.shape)

train=[wdl.split() for wdl in train[column]]
test=[wdl.split() for wdl in test[column]]
maxlen=max([len(x) for x in test+train])
print("maxlen:",maxlen)

def w2v_pad(data):
	data = [[wv[s] for s in st if s in wv.vocab] for st in data]
	data = sequence.pad_sequences(data, maxlen=maxlen)
	print('data shape:', data.shape)

train=w2v_pad(train)
test=w2v_pad(test)

t2=time.time()
print("dataproc time use:",t2-t1)

model = Sequential()
model.add(LSTM(64, input_shape=(train.shape[1], train.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(train, y,batch_size=1024,epochs=100,validation_split=0.2,verbose=2)
preds=model.predict(test,batch_size=1024,verbose=1)

t3=time.time()
print("lstm time use:",t3-t2)

test_prob=pd.DataFrame(preds)
test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
test_prob["id"]=test_id
test_prob.to_csv('../data/w2v+lstm_prob.csv',index=None)

test_pred=pd.DataFrame(np.argmax(preds,axis=1))
test_pred.columns=["class"]
test_pred["class"]=(test_pred["class"]+1).astype(int)
test_pred["id"]=test_id
test_pred[["id","class"]].to_csv('../data/w2v+lstm_sub.csv',index=None)

print("probs shape:",test_prob.shape)
print("preds shape:",test_pred.shape)