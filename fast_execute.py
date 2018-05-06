import numpy as np
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
import keras.layers as layers
from keras.layers import GRU, LSTM, SimpleRNN
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)

from keras.callbacks import ModelCheckpoint

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

print('\n\n\n\n\n\n\n\n\n\n\n')

print('#########################################################\n#                                                       #\n#       Demonstration video: 3i for Korean (3i4K)       #\n#                                                       #\n#########################################################')

print('\nPreparing data...')

dataset_fcix = read_data('fci_x.txt')
dataset_fciy_ =read_data('fci_y.txt')
dataset_fciy = np.zeros(len(dataset_fciy_))
for i in range(len(dataset_fciy)):
  dataset_fciy[i] = float(dataset_fciy_[i][0])

######## 3-class

dataset_3x = read_data('3class_x.txt')
dataset_3y_ = read_data('3class_y.txt')
dataset_3y = np.zeros(len(dataset_3y_))
for i in range(len(dataset_3y)):
  dataset_3y[i] = float(dataset_3y_[i][0])

######## 5-class 

dataset_5x = read_data('5class_x.txt')
dataset_5y_ = read_data('5class_y.txt')
dataset_5y = np.zeros(len(dataset_5y_))
for i in range(len(dataset_5y)):
  dataset_5y[i] = float(dataset_5y_[i][0])

import nltk
from konlpy.tag import Twitter
pos_tagger = Twitter()

print('Tokenizing...')

def twit_token(doc):
    x = [t[0] for t in pos_tagger.pos(doc)]
    return ' '.join(x)

def twit_pos(doc):
    x = [t[1] for t in pos_tagger.pos(doc)]
    return ' '.join(x)

#fci_token = [(twit_token(str(row)[2:-2])) for row in dataset_fcix]
#fci_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_fcix]

#data3_token = [(twit_token(str(row)[2:-2])) for row in dataset_3x]
#data3_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_3x]

#data5_token = [(twit_token(str(row)[2:-2])) for row in dataset_5x]
#data5_pos   = [(twit_pos(str(row)[2:-2])) for row in dataset_5x]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
'''
corpus1 = data3_token
corpus2 = data3_pos

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

#fci_tfidf,fci_tfidf_bi, fci_postfidf,fci_postfidf_bi,fci_posindex = tfidf_featurizer(fci_token,fci_pos)
#data3_tfidf,data3_tfidf_bi, data3_postfidf,data3_postfidf_bi, data3_posindex = tfidf_featurizer(data3_token,data3_pos)
#data5_tfidf,data5_tfidf_bi, data5_postfidf,data5_postfidf_bi, data5_posindex = tfidf_featurizer(data5_token,data5_pos)
'''
print('Importing word vectors...')
import fasttext

model_ft = fasttext.load_model('model.bin')

print('Importing models...\n')

from keras.models import load_model

model3_bilstm = load_model('model3/bilstm-17-0.8843.hdf5')
model5_bilstm = load_model('model5/bilstm-16-0.8750.hdf5')
modelfci_bilstm = load_model('modelfci/cnn-16-0.8176.hdf5')
modelaux_bilstm = load_model('modelaux/bilstm32-13-0.8695.hdf5')
modelnaux_bilstm = load_model('modelnaux/bilstm32-10-0.8543.hdf5')

maxlen = 20
wdim   =100

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

def predict_3(st):
    token = twit_token(st)
    pos   = twit_pos(st)
    s    = nltk.word_tokenize(token)
    spos = nltk.word_tokenize(pos)
    rec = np.zeros((1,maxlen,wdim))
    for j in range(len(s)):
         if s[j] in model_ft and j<maxlen:
            rec[0][j,:]=model_ft[s[j]]
    rec1 = model3_bilstm.predict(rec)
    return rec1

def predict_5(st):
    token = twit_token(st)
    pos   = twit_pos(st)
    s    = nltk.word_tokenize(token)
    spos = nltk.word_tokenize(pos)
    rec = np.zeros((1,maxlen,wdim))
    for j in range(len(s)):
         if s[j] in model_ft and j<maxlen:
            rec[0][j,:]=model_ft[s[j]]
    rec1 = model5_bilstm.predict(rec)
    return rec1

def predict_intention(st):
  z = predict_fci(st)
  c = np.argmax(z[0])
  if c==0:
    print('끝까지 말씀해 주세요.\n')
    return 0
  elif c==2:
    print('부가 정보가 필요합니다.\n')
    return 1
  else:  
    y = predict_3(st)
    d = np.argmax(y[0])
    if d==0:
        print('...(침묵)\n')
        return 2
    elif d==1:
        print('그 점이 궁금하셨나요?\n')
        return 3
    else:
        print('요청 접수되었습니다.\n')
        return 4

def predict_intention_5(st):
  z = predict_fci(st)
  c = np.argmax(z[0])
  if c==0:
    print('Not enough information...\n')
    return -2
  elif c==2:
    print('Intonation information required.\n')
    return -1
  else:
    y = predict_5(st)
    d = np.argmax(y[0])
    if d==0:
        print('This question is rhetorical...\n')
    elif d==1:
        print('This command is rhetorical...\n')
    elif d==2:
        print('Statement.\n')
    elif d==3:
        print('Question?\n')
    elif d==4:
        print('Command!\n')
    return d

def predict_intention_aux(st):
  token = twit_token(st)
  pos   = twit_pos(st)
  s    = nltk.word_tokenize(token)
  spos = nltk.word_tokenize(pos)
  rec = np.zeros((1,maxlen,wdim))
  for j in range(len(s)):
    if s[j] in model_ft and j<maxlen:
      rec[0][j,:]=model_ft[s[j]]
  ###################
  z = predict_fci(st)
  c = np.argmax(z[0])
  aux2=np.zeros((1,3))
  for j in range(3):
    aux2[0][j] = z[0][j]
  if c==0:
    print('Not enough information...\n')
  else:
   if c==1:
    #aux1_=0
    rec1=model5_bilstm.predict([rec])
    d = np.argmax(rec1[0])
    if d==0:
        print('This question is rhetorical...\n')
    elif d==1:
        print('This command is rhetorical...\n')
    elif d==2:
        print('Statement.\n')
    elif d==3:
        print('Question?\n')
    elif d==4:
        print('Command!\n')
   else: 
    aux1_=input('Intonation information required.\n 1: High rise 2: Low rise 3: Fall-rise 4: Level 5: Fall\n')
    #if isinstance(aux1_,int)==True and int(aux1_)<=5:
    if len(aux1_)==1 and any(c.isdigit() for c in aux1_):
     if int(aux1_)>5:
      print('Please insert correct label number.')
     else:
      aux1_=int(aux1_)
      aux1=np.zeros((1,6))
      aux1[0][int(aux1_)]=1
      aux=np.concatenate((aux1,aux2),axis=1)
      auxrec1=modelnaux_bilstm.predict([rec,aux])
      d = np.argmax(auxrec1[0])
      if d==0:
         print('This question is rhetorical...\n')
      elif d==1:
         print('This command is rhetorical...\n')
      elif d==2:
         print('Statement.\n')
      elif d==3:
         print('Question?\n')
      elif d==4:
         print('Command!\n')

def classify_document(filename):
  target3 = read_data(filename+'.txt')
  f0 = open(filename+'_frag.txt','w')
  f1 = open(filename+'_into.txt','w')
  f2 = open(filename+'_dec.txt','w')
  f3 = open(filename+'_int.txt','w')
  f4 = open(filename+'_imp.txt','w')
  for i in range(len(target3)):
    d = predict_intention(target3[i][0])
    if i%2000 == 0:
        print(i,'over',len(target3))
    if d == 0:
        f0.write(target3[i][0]+'\n')
    if d == 1:
        f1.write(target3[i][0]+'\n')
    if d == 2:
        f2.write(target3[i][0]+'\n')
    elif d==3:
        f3.write(target3[i][0]+'\n')
    else:
        f4.write(target3[i][0]+'\n')
  f0.close()
  f1.close()
  f2.close()
  f3.close()
  f4.close()

def classify_document_5(filename):
  target5 = read_data(filename+'.txt')
  ff = open(filename+'_frag.txt','w')
  fi = open(filename+'_into.txt','w')
  f0 = open(filename+'_rq.txt','w')
  f1 = open(filename+'_ie.txt','w')
  f2 = open(filename+'_dec.txt','w')
  f3 = open(filename+'_int.txt','w')
  f4 = open(filename+'_imp.txt','w')
  for i in range(len(target5)):
    print(target5[i][0])
    d = predict_intention_5(target5[i][0])
    if i%2000 == 0:
        print(i,'over',len(target5))
    if d == -2:
        ff.write(target5[i][0]+'\n')
    if d == -1:
        fi.write(target5[i][0]+'\n')
    if d == 0:
        f0.write(target5[i][0]+'\n')
    if d == 1:
        f1.write(target5[i][0]+'\n')
    if d == 2:
        f2.write(target5[i][0]+'\n')
    elif d==3:
        f3.write(target5[i][0]+'\n')
    else:
        f4.write(target5[i][0]+'\n')
  ff.close()
  fi.close()
  f0.close()
  f1.close()
  f2.close()
  f3.close()
  f4.close()

print('#########################################################\n\n  Intention identification and text classification demo\n\n#########################################################\n\nEnter "i" to identify a sentence\nEnter "c" to classify a corpus\nEnter "bye" to quit\n')

from pathlib import Path

def detect():
  print('Identification demo ...\nEnter "c" to activate classification mode\nEnter "bye" to quit\n')
  while 1:
    s = input('You say: ')
    if s == 'bye':
        sys.exit() 
    elif s == 'c':
        classify()
    else:
        predict_intention_aux(s)

def classify():
  print('Classification demo ...\nEnter "i" to activate identification mode\nEnter "bye" to quit\n')
  while 1:
    s = input('Target filename (.txt): ')
    if s == 'bye':
        sys.exit()
    elif s == 'i':
        detect()
    else:
      if Path(s+'.txt').is_file():
        classify_document_5(s)
      else:
        print('No such file exists!')

while 1:
  s = input('Choose: ')
  if s == 'i':
    detect()
  elif s == 'c':
    classify()
  elif s == 'bye':
    sys.exit() 
