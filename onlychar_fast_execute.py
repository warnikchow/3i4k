import numpy as np
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
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

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

print('\n\n\n\n\n\n\n\n\n\n\n')

print('#########################################################\n#                                                       #\n#       Demonstration video: 3i for Korean (3i4K)       #\n#                                                       #\n#########################################################')

print('\nPreparing data...')

print('Tokenizing...')

print('Importing word vectors...')
import fasttext

model_ft = fasttext.load_model('model.bin')

print('Importing models...\n')

from keras.models import load_model

model5_cnn_fusion = load_model('model/model5/cnn_char_bilstm_char_space-15-0.9024-F0.7930.hdf5')
modelfci_bilstm = load_model('model/modelfci/cnn_bilstm_char_fusion-16-0.8679-F0.8100.hdf5')
modelnaux_bilstm = load_model('model/modelnaux/bilstm128-17-0.8938.hdf5')

maxcharlen = 50
wdim   =100

################################

def predict_fci(st):
    rec = np.zeros((1,maxcharlen,wdim))
    conv = np.zeros((1,maxcharlen,wdim,1))
    countchar=-1
    for j in range(len(st)):
        if st[j] in model_ft and j<maxcharlen:
            rec[0][j,:]=model_ft[st[j]]
            conv[0][j,:,0]=model_ft[st[j]]
    rec1 = modelfci_bilstm.predict([rec,conv])
    return rec1

def predict_5(st):
    rec = np.zeros((1,maxcharlen,wdim))
    conv = np.zeros((1,maxcharlen,wdim,1))
    for j in range(len(st)):
         if st[j] in model_ft and j<maxcharlen:
            rec[0][j,:]=model_ft[st[j]]
            conv[0][j,:,0]=model_ft[st[j]]
    rec1 = model5_cnn_fusion.predict([rec,conv])
    return rec1

################################

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

def predict_intention_5kor(st):
  z = predict_fci(st)
  c = np.argmax(z[0])
  if c==0:
    print('부가 정보가 필요합니다...\n')
    return -2
  elif c==2:
    print('판단이 확실하지 않네요.\n')
    return -1
  else:
    y = predict_5(st)
    d = np.argmax(y[0])
    if d==0:
        print('진심으로 궁금하신 건 아니죠?\n')
    elif d==1:
        print('그 요청은 따로 수행하지 않겠습니다.\n')
    elif d==2:
        print('그렇군요.\n')
    elif d==3:
        print('궁금한 점이 있으신 것 같네요.\n')
    elif d==4:
        print('요청 접수되었습니다.\n')
    return d

################################

def predict_intention_aux(st):
  rec = np.zeros((1,maxcharlen,wdim))
  for j in range(len(st)):
    if st[j] in model_ft and j<maxcharlen:
      rec[0][j,:]=model_ft[st[j]]
  conv2 = np.zeros((1,maxcharlen,wdim,1))
  for j in range(len(st)):
        if st[j] in model_ft and j<maxcharlen:
            conv2[0][j,:,0]=model_ft[st[j]]
  ###################
  z = predict_fci(st)
  c = np.argmax(z[0])
  aux2=np.zeros((1,3))
  for j in range(3):
    aux2[0][j] = z[0][j]
  if c==0:
    print(' >> Not enough information...\n')
  else:
   if c==1:
    #aux1_=0
    rec1=model5_cnn_fusion.predict([rec,conv2])
    d = np.argmax(rec1[0])
    if d==0:
        print(' >> This question is rhetorical...\n')
    elif d==1:
        print(' >> This command is rhetorical...\n')
    elif d==2:
        print(' >> Statement.\n')
    elif d==3:
        print(' >> Question?\n')
    elif d==4:
        print(' >> Command!\n')
   else: 
    aux1_=input(' >> Intonation information required.\n 1: High rise 2: Low rise 3: Fall-rise 4: Level 5: Fall\n  Auxiliary input: ')
    #if isinstance(aux1_,int)==True and int(aux1_)<=5:
    if len(aux1_)==1 and any(c.isdigit() for c in aux1_):
     if int(aux1_)>5:
      print(' >> Please insert correct label number.\n')
     else:
      aux1_=int(aux1_)
      aux1=np.zeros((1,6))
      aux1[0][int(aux1_)]=1
      aux=np.concatenate((aux1,aux2),axis=1)
      auxrec1=modelnaux_bilstm.predict([rec,aux])
      d = np.argmax(auxrec1[0])
      if d==0:
         print(' >> This question is rhetorical...\n')
      elif d==1:
         print(' >> This command is rhetorical...\n')
      elif d==2:
         print(' >> Statement.\n')
      elif d==3:
         print(' >> Question?\n')
      elif d==4:
         print(' >> Command!\n')

def predict_intention_auxkr(st):
  rec = np.zeros((1,maxcharlen,wdim))
  for j in range(len(st)):
    if st[j] in model_ft and j<maxcharlen:
      rec[0][j,:]=model_ft[st[j]]
  conv2 = np.zeros((1,maxcharlen,wdim,1))
  for j in range(len(st)):
        if st[j] in model_ft and j<maxcharlen:
            conv2[0][j,:,0]=model_ft[st[j]]
  ###################
  z = predict_fci(st)
  c = np.argmax(z[0])
  aux2=np.zeros((1,3))
  for j in range(3):
    aux2[0][j] = z[0][j]
  if c==0:
    print(' >> 부가 정보가 필요합니다...\n')
  else:
   if c==1:
    #aux1_=0
    rec1=model5_cnn_fusion.predict([rec,conv2])
    d = np.argmax(rec1[0])
    if d==0:
        print(' >> 진심으로 궁금하신 건 아니죠?\n')
    elif d==1:
        print(' >> 그 요청은 따로 수행하지 않겠습니다.\n')
    elif d==2:
        print(' >> 그렇군요.\n')
    elif d==3:
        print(' >> 궁금한 점이 있으신 것 같네요.\n')
    elif d==4:
        print(' >> 요청 접수되었습니다.\n')
   else:
    aux1_=input('\n >> 판단이 확실치 않네요.\n >> 문말 억양 정보를 입력해 주세요\n\n >> 1: High rise \n >> 2: Low rise \n >> 3: Fall-rise \n >> 4: Level tone \n >> 5: Fall \n\n  억양 번호 입력: ')
    #if isinstance(aux1_,int)==True and int(aux1_)<=5:
    if len(aux1_)==1 and any(c.isdigit() for c in aux1_):
     if int(aux1_)>5:
      print(' >> 올바른 번호를 입력해 주세요.\n')
     else:
      aux1_=int(aux1_)
      aux1=np.zeros((1,6))
      aux1[0][int(aux1_)]=1
      aux=np.concatenate((aux1,aux2),axis=1)
      auxrec1=modelnaux_bilstm.predict([rec,aux])
      d = np.argmax(auxrec1[0])
      if d==0:
         print(' >> 진심으로 궁금하신 건 아니죠?\n')
      elif d==1:
         print(' >> 그 요청은 따로 수행하지 않겠습니다.\n')
      elif d==2:
         print(' >> 그렇군요.\n')
      elif d==3:
         print(' >> 궁금한 점이 있으신 것 같네요.\n')
      elif d==4:
         print(' >> 요청 접수되었습니다.\n')

################################

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

print('#########################################################\n\n  Intention identification and text classification demo\n\n#########################################################\n\nEnter "i" to identify a sentence\nEnter "c" to classify a corpus\nEnter "bye" to quit\n\n')

from pathlib import Path

def detect():
  print('Identification demo ...\nEnter "c" to activate classification mode\nEnter "bye" to quit\n')
  while 1:
    s = input(' You say: ')
    if s == 'bye':
        sys.exit() 
    elif s == 'c':
        classify()
    else:
        predict_intention_auxkr(s)

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
  s = input(' Choose: ')
  if s == 'i':
    detect()
  elif s == 'c':
    classify()
  elif s == 'bye':
    sys.exit() 
