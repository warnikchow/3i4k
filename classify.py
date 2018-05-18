import numpy as np
import sys
import codecs

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

from random import shuffle
######## frag/clear-cut/ID

dataset_fcix = read_data('data/fci_x.txt')
dataset_fciy_ =read_data('data/fci_y.txt')
dataset_fciy = np.zeros(len(dataset_fciy_))
for i in range(len(dataset_fciy)):
  dataset_fciy[i] = float(dataset_fciy_[i][0])

####### fci_re
dataset_fci = read_data('data/fci_re.txt')

######## 3-class

dataset_3x = read_data('data/3class_x.txt')
dataset_3y_ = read_data('data/3class_y.txt')
dataset_3y = np.zeros(len(dataset_3y_))
for i in range(len(dataset_3y)):
  dataset_3y[i] = float(dataset_3y_[i][0])

######## 5-class 

dataset_5x = read_data('data/5class_x.txt')
dataset_5y_ = read_data('data/5class_y.txt')
dataset_5y = np.zeros(len(dataset_5y_))
for i in range(len(dataset_5y)):
  dataset_5y[i] = float(dataset_5y_[i][0])

######## into_dep

dataset_intox = read_data('data/intodep_x.txt')
dataset_intoaux_ = read_data('data/intodep_aux.txt')
dataset_intoy_ = read_data('data/intodep_y.txt')
dataset_intoaux = np.zeros(len(dataset_intoaux_))
dataset_intoy = np.zeros(len(dataset_intoy_))
for i in range(len(dataset_intoaux)):
  dataset_intoaux[i] = float(dataset_intoaux_[i][0])
  dataset_intoy[i] = float(dataset_intoy_[i][0])

######## intodep_only

dataset_nintox = read_data('data/intodep_only_x.txt')
dataset_nintoaux_ = read_data('data/intodep_only_aux.txt')
dataset_nintoy_ = read_data('data/intodep_only_y.txt')
dataset_nintoaux = np.zeros(len(dataset_nintoaux_))
dataset_nintoy = np.zeros(len(dataset_nintoy_))
for i in range(len(dataset_nintoaux)):
  dataset_nintoaux[i] = float(dataset_nintoaux_[i][0])
  dataset_nintoy[i] = float(dataset_nintoy_[i][0])

import fasttext
model_ft = fasttext.load_model('model.bin')

import nltk
from konlpy.tag import Twitter
pos_tagger = Twitter()

print('Tokenizing...to TF-IDF features')

def twit_token(doc):
    x = [t[0] for t in pos_tagger.pos(doc)]
    return ' '.join(x)

def twit_pos(doc):
    x = [t[1] for t in pos_tagger.pos(doc)]
    return ' '.join(x)

fci_token = [(twit_token(str(row)[2:-2])) for row in dataset_fcix]
fci_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_fcix]

####### fci_re
fci_token = [(twit_token(str(row[1]))) for row in dataset_fci]
fci_pos   = [(twit_pos(str(row[1]))) for row in dataset_fci]
dataset_fciy = [row[0] for row in dataset_fci]

data3_token = [(twit_token(str(row)[2:-2])) for row in dataset_3x]
data3_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_3x]

data5_token = [(twit_token(str(row)[2:-2])) for row in dataset_5x]
data5_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_5x]

datainto_token = [(twit_token(str(row)[2:-2])) for row in dataset_intox]
datainto_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_intox]

dataninto_token = [(twit_token(str(row)[2:-2])) for row in dataset_nintox]
dataninto_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_nintox]

# https://gist.github.com/jason-riddle/1a854af26562c0cdb1e6ff550d1bf32d#file-complex-tf-idf-example-py-L40
# http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/

#from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_featurizer(corpus1,corpus2):
  count_vectorizer = CountVectorizer()
  count_vectorizer.fit_transform(corpus2)
  freq_term_matrix = count_vectorizer.transform(corpus2)
  tfidf = TfidfTransformer(norm="l2")
  tfidf.fit(freq_term_matrix)
  tfidf_token = TfidfVectorizer(ngram_range=(1,1),max_features=1000)
  token_tfidf_total = tfidf_token.fit_transform(corpus1)
  token_tfidf_total_mat = token_tfidf_total.toarray()
  tfidf_pos = TfidfVectorizer(ngram_range=(1,1))
  pos_tfidf_total = tfidf_pos.fit_transform(corpus2)
  pos_tfidf_total_mat = pos_tfidf_total.toarray()
  pos_index={i[1]:i[0] for i in tfidf_pos.vocabulary_.items()}
  pos_indexed = []
  pos_tfidf_total = np.array(pos_tfidf_total.todense())
  for row in pos_tfidf_total:
    pos_indexed.append({pos_index[column]:value for (column,value) in enumerate(row)})
  tfidf_bi_token = TfidfVectorizer(ngram_range=(1,2),max_features=1000)
  token_tfidf_bi_total = tfidf_bi_token.fit_transform(corpus1)
  token_tfidf_bi_total_mat = token_tfidf_bi_total.toarray()
  tfidf_bi_pos = TfidfVectorizer(ngram_range=(1,2))
  pos_tfidf_bi_total = tfidf_bi_pos.fit_transform(corpus2)
  pos_tfidf_bi_total_mat = pos_tfidf_bi_total.toarray()
  return token_tfidf_total_mat,token_tfidf_bi_total_mat,pos_tfidf_total_mat,pos_tfidf_bi_total_mat, pos_indexed

fci_tfidf,fci_tfidf_bi, fci_postfidf,fci_postfidf_bi,fci_posindex = tfidf_featurizer(fci_token,fci_pos)
data3_tfidf,data3_tfidf_bi, data3_postfidf,data3_postfidf_bi, data3_posindex = tfidf_featurizer(data3_token,data3_pos)
data5_tfidf,data5_tfidf_bi, data5_postfidf,data5_postfidf_bi, data5_posindex = tfidf_featurizer(data5_token,data5_pos)

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, fci_tfidf, dataset_fciy, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_fciy, predicted)
f1_score = metrics.f1_score(dataset_fciy, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, fci_tfidf_bi, dataset_fciy, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_fciy, predicted)
f1_score = metrics.f1_score(dataset_fciy, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, fci_postfidf_bi, dataset_fciy, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_fciy, predicted)
f1_score = metrics.f1_score(dataset_fciy, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, data3_tfidf, dataset_3y, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_3y, predicted)
f1_score = metrics.f1_score(dataset_3y, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, data3_tfidf_bi, dataset_3y, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_3y, predicted)
f1_score = metrics.f1_score(dataset_3y, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, data3_postfidf_bi, dataset_3y, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_3y, predicted)
f1_score = metrics.f1_score(dataset_3y, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, data5_tfidf, dataset_5y, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_5y, predicted)
f1_score = metrics.f1_score(dataset_5y, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, data5_tfidf_bi, dataset_5y, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_5y, predicted)
f1_score = metrics.f1_score(dataset_5y, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

CLF = LinearSVC()
K_FOLDS=10
predicted = cross_val_predict(CLF, data5_postfidf_bi, dataset_5y, cv=K_FOLDS) # training된 시스템으로 prediction
acc_score= metrics.accuracy_score(dataset_5y, predicted)
f1_score = metrics.f1_score(dataset_5y, predicted, average="macro") # prediction의 정확도를 f1 score의 관점에서 측정
print('acc: ', acc_score)
print('f1: ', f1_score)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
adam_half_2 = optimizers.Adam(lr=0.0002)

from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding

from random import random
from numpy import array
from numpy import cumsum
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from keras.callbacks import ModelCheckpoint

from keras.layers.normalization import BatchNormalization

hidden = 64
wdim   = 100

def featurize_cnn(corpus,maxlen):
  conv_total = np.zeros((len(corpus),maxlen,wdim,1))
  for i in range(len(corpus)):
    if i%1000 == 0:
        print(i)
    s = nltk.word_tokenize(corpus[i])
    for j in range(len(s)):
        if s[j] in model_ft and j<maxlen:
          conv_total[i][j,:,0]=model_ft[s[j]]
  return conv_total

def featurize_charcnn(corpus,maxlen):
  conv_total = np.zeros((len(corpus),maxlen,wdim,1))
  for i in range(len(corpus)):
    if i%1000 ==0:
      print(i)
    s = corpus[i]
    for j in range(len(s)):
      if s[j] in model_ft and j<maxlen:
          conv_total[i][j,:,0]=model_ft[s[j]]
  return conv_total

fci_conv = featurize_cnn(fci_token,20)
data3_conv = featurize_cnn(data3_token,20)
data5_conv = featurize_cnn(data5_token,20)
datainto_conv = featurize_cnn(datainto_token,20)
dataninto_conv = featurize_cnn(dataninto_token,20)

fci_conv_char = featurize_charcnn(fci_token,50)
data5_conv_char =featurize_charcnn(data5_token,50)

def validate_cnn(result,y,label_num,filename):
  model = Sequential()
  model.add(layers.Conv2D(hidden,(3,wdim),activation= 'relu',input_shape = (len(result[0]),wdim,1)))
  model.add(layers.MaxPooling2D((2,1)))
  model.add(layers.Conv2D(hidden,(3,1),activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(hidden,activation='relu'))
  model.add(Dense(label_num, activation='softmax'))
  model.summary()
  model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  model.fit(result,y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

validate_cnn(fci_conv,dataset_fciy,3,'model/modelfci/cnn')
validate_cnn(data3_conv,dataset_3y,3,'model/model3/cnn')
validate_cnn(data5_conv,dataset_5y,5,'model/model5/cnn')

validate_cnn(fci_conv_char,dataset_fciy,3,'model/modelfci/charcnn')
validate_cnn(data5_conv_char,dataset_5y,5,'model/model5/charcnn')

def validate_cnn_fusion(x_cnn,x_char,x_y,filename):
  word_seq0 = Input(shape=(len(x_cnn[0]),wdim,1),dtype='float32')
  word_seq = layers.Conv2D(hidden,(3,wdim),activation= 'relu')(word_seq0)
  word_seq = layers.MaxPooling2D((2,1))(word_seq)
  word_seq = layers.Conv2D(hidden,(3,1),activation='relu')(word_seq)
  word_seq = layers.Flatten()(word_seq)
  char_seq0 = Input(shape=(len(x_char[0]),wdim,1),dtype='float32')
  char_seq = layers.Conv2D(hidden,(3,wdim),activation= 'relu')(char_seq0)
  char_seq = layers.MaxPooling2D((2,1))(char_seq)
  char_seq = layers.Conv2D(hidden,(3,1),activation='relu')(char_seq)
  char_seq = layers.Flatten()(char_seq)
  concat = layers.concatenate([word_seq, char_seq]) 
  concat = Dense(hidden, activation='relu')(concat)
  main_output = Dense(int(max(x_y))+1,activation='softmax')(concat)
  model = Sequential()
  model = Model(inputs=[word_seq0,char_seq0],outputs=[main_output])
  model.summary()
  model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
  filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  model.fit([x_cnn,x_char],x_y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

validate_cnn_fusion(fci_conv,fci_conv_char,dataset_fciy,'model/modelfci/cnn_fusion')

def featurize_rnn(corpus,maxlen):
  rec_total = np.zeros((len(corpus),maxlen,wdim))
  for i in range(len(corpus)):
    if i%1000 ==0:
        print(i)
    s = nltk.word_tokenize(corpus[i])
    for j in range(len(s)):
        if s[j] in model_ft and j<maxlen:
            rec_total[i][j,:]=model_ft[s[j]]
  return rec_total

def featurize_charrnn(corpus,maxlen):
  rec_total = np.zeros((len(corpus),maxlen,wdim))
  for i in range(len(corpus)):
    if i%1000 ==0:
        print(i)
    s = corpus[i]
    for j in range(len(s)):
        if s[j] in model_ft and j<maxlen:
            rec_total[i][j,:]=model_ft[s[j]]
  return rec_total

fci_rec   = featurize_rnn(fci_token,20)
data3_rec = featurize_rnn(data3_token,20)
data5_rec = featurize_rnn(data5_token,20)
datainto_rec = featurize_rnn(datainto_token,20)
dataninto_rec = featurize_rnn(dataninto_token,20)

fci_rec_char = featurize_charrnn(fci_token,50)

def validate_bilstm(result,y,label_num,filename):
  model = Sequential()
  model.add(Bidirectional(LSTM(32), input_shape=(len(result[0]), wdim)))
  model.add(Dense(hidden, activation='relu'))
  model.add(Dense(label_num, activation='softmax'))
  model.summary()
  model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
  filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  model.fit(result,y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

validate_bilstm(fci_rec,dataset_fciy,3,'model/modelfci/bilstm')
validate_bilstm(data3_rec,dataset_3y,3,'model/model3/bilstm')
validate_bilstm(data5_rec,dataset_5y,5,'model/model5/bilstm')

def validate_rnn_fusion(x_rnn,x_char,x_y,filename):
  word_seq0 = Input(shape=(len(x_rnn[0]),wdim),dtype='float32')
  word_seq = Bidirectional(LSTM(32))(word_seq0)
  char_seq0 = Input(shape=(len(x_char[0]),wdim,1),dtype='float32')
  char_seq = layers.Conv2D(hidden,(3,wdim),activation= 'relu')(char_seq0)
  char_seq = layers.MaxPooling2D((2,1))(char_seq)
  char_seq = layers.Conv2D(hidden,(3,1),activation='relu')(char_seq)
  char_seq = layers.Flatten()(char_seq)
  concat = layers.concatenate([word_seq, char_seq])
  concat = Dense(hidden, activation='relu')(concat)
  main_output = Dense(int(max(x_y))+1,activation='softmax')(concat)
  model = Sequential()
  model = Model(inputs=[word_seq0,char_seq0],outputs=[main_output])
  model.summary()
  model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
  filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  model.fit([x_rnn,x_char],x_y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

validate_rnn_fusion(data5_rec,data5_conv_char,dataset_5y,'model/model5/bilstm_fusion')

'''
classify_document('topics/Email')
classify_document('topics/HouseControl')
classify_document('topics/Schedule')
classify_document('topics/Weather')

classify_document('topics_v7/Email')
classify_document('topics_v7/HouseControl')
classify_document('topics_v7/Schedule')
classify_document('topics_v7/Weather')
'''

from keras.models import load_model

model3_bilstm = load_model('model/model3/bilstm-17-0.8843.hdf5')
modelfci_bilstm = load_model('model/modelfci/cnn-16-0.8176.hdf5')

def predict_fci(st):
    token = twit_token(st)
    pos = twit_pos(st)
    s = nltk.word_tokenize(token)
    spos = nltk.word_tokenize(pos)
    conv = np.zeros((1,maxlen,wdim,1))
    for j in range(len(s)):
        if s[j] in model_ft and j<maxlen:
            conv[0][j,:,0]=model_ft[s[j]]
    conv1 = modelfci_bilstm.predict(conv)
    return conv1

def make_fci_predict(corpus):
    result = np.zeros((len(corpus),3))
    for i in range(len(corpus)):
      if i%1000 ==0:
        print(i)
      for j in range(3):
        z = predict_fci(str(corpus[i])[2:-2])
        result[i][j] = z[0][j]
    return result

def make_into_label(corpus):
    result = np.zeros((len(corpus),int(max(corpus))+1))
    for i in range(len(corpus)):
      if i%1000 ==0:
        print(i)
      result[i][int(corpus[i])]=1
    return result
    
dataset_intofci=make_fci_predict(dataset_intox)
dataset_intolabel=make_into_label(dataset_intoaux)
dataset_finalaux=np.concatenate((dataset_intolabel,dataset_intofci),axis=1)

dataset_nintofci=make_fci_predict(dataset_nintox)
dataset_nintolabel=make_into_label(dataset_nintoaux)
dataset_nfinalaux=np.concatenate((dataset_nintolabel,dataset_nintofci),axis=1)

maxlen = 20

def aux_bilstm(x_rnn,x_finalaux,x_y,hidden_dim,filename):
  word_seq = Input(shape=(maxlen,wdim),dtype='float32')
  word_hidden = Bidirectional(LSTM(32,return_sequences=True))(word_seq)
  aux_input = Input(shape=(len(x_finalaux[0]),),dtype='float32')
  aux_lstm  = Dense(hidden_dim, activation = 'relu')(aux_input)
  aux_lstm  = Dense(hidden_dim, activation = 'relu')(aux_lstm)
  aux_attention = Dense(maxlen, activation = 'softmax')(aux_lstm)
  aux_attention = keras.layers.Reshape((maxlen,1))(aux_attention)
  word_aux_sum = keras.layers.multiply([aux_attention,word_hidden])
  word_aux_sum = Lambda(lambda x: K.sum(x, axis=1))(word_aux_sum)
  main_output = Dense(int(max(x_y))+1,activation='softmax')(word_aux_sum)
  model = Sequential()
  model = Model(inputs=[word_seq,aux_input],outputs=[main_output])
  model.summary()
  model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
  filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]
  model.fit([x_rnn,x_finalaux],x_y,validation_split=0.1,epochs=20,batch_size=16,callbacks=callbacks_list)

aux_bilstm(datainto_rec,dataset_finalaux,dataset_intoy,32,'model/modelaux/bilstm32')
aux_bilstm(datainto_rec,dataset_finalaux,dataset_intoy,64,'model/modelaux/bilstm64')
aux_bilstm(datainto_rec,dataset_finalaux,dataset_intoy,128,'model/modelaux/bilstm128')

aux_bilstm(dataninto_rec,dataset_nfinalaux,dataset_nintoy,32,'model/modelnaux/bilstm32')
aux_bilstm(dataninto_rec,dataset_nfinalaux,dataset_nintoy,64,'model/modelnaux/bilstm64')
aux_bilstm(dataninto_rec,dataset_nfinalaux,dataset_nintoy,128,'model/modelnaux/bilstm128')
