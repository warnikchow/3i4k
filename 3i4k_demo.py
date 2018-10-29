import numpy as np
import sys

print('\n\n\n\n\n\n\n\n\n\n\n')

print('#########################################################\n#                                                       #\n#       Demonstration video: 3i for Korean (3i4K)       #\n#                                                       #\n#########################################################')

import fasttext

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

model_ft = fasttext.load_model('vectors/model_drama.bin')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))

from keras.models import load_model
import librosa

print('Importing models...\n')

model_fci  = load_model('model/modelfci/charbilstm_self_rere-23-0.8828.hdf5')
mse_crs  = load_model('model/total_s_rmse_cnn_rnnself-12-0.7714-f0.4412.hdf5')
model_aux=load_model('model/modelaux/bilstm_att_re-29-0.9016.hdf5')

mlen=300
wdim=100

def featurize_charrnn_utt(corpus,maxcharlen):
  rec_total = np.zeros((1,maxcharlen,wdim))
  s = corpus
  for j in range(len(s)):
        if s[-j-1] in model_ft and j<maxcharlen:
            rec_total[0,-j-1,:]=model_ft[s[-j-1]]
  return rec_total

def featurize_charcnn_utt(corpus,maxcharlen):
  conv_total = np.zeros((1,maxcharlen,wdim,1))
  s = corpus
  for j in range(len(s)):
        if s[-j-1] in model_ft and j<maxcharlen:
            conv_total[0,-j-1,:,0]=model_ft[s[-j-1]]
  return conv_total

def make_data(filename):
  data=np.zeros((1,mlen,128))
  data_conv=np.zeros((1,mlen,128,1))
  data_rmse=np.zeros((1,mlen,1))
  data_srmse=np.zeros((1,mlen,128))
  data_srmse_conv=np.zeros((1,mlen,128,1))
  data_s_rmse=np.zeros((1,mlen,129))
  data_s_rmse_conv=np.zeros((1,mlen,129,1))
  y, sr = librosa.load(filename)
  D = np.abs(librosa.stft(y))**2
  ss, phase = librosa.magphase(librosa.stft(y))
  rmse = librosa.feature.rmse(S=ss)
  rmse = rmse/np.max(rmse)
  rmse = np.transpose(rmse)
  S = librosa.feature.melspectrogram(S=D)
  S = np.transpose(S)
  Srmse = np.multiply(rmse,S)
  if len(S)>=mlen:
    data[0][:,:]=S[-mlen:,:]
    data_conv[0][:,:,0]=S[-mlen:,:]
    data_rmse[0][:,0]=rmse[-mlen:,0]
    data_srmse[0][:,:]=Srmse[-mlen:,:]
    data_srmse_conv[0][:,:,0]=Srmse[-mlen:,:]
    data_s_rmse[0][:,0]=rmse[-mlen:,0]
    data_s_rmse[0][:,1:]=S[-mlen:,:]
    data_s_rmse_conv[0][:,0,0]=rmse[-mlen:,0]
    data_s_rmse_conv[0][:,1:,0]=S[-mlen:,:]
  else:
    data[0][-len(S):,:]=S
    data_conv[0][-len(S):,:,0]=S
    data_rmse[0][-len(S):,0]=np.transpose(rmse)
    data_srmse[0][-len(S):,:]=Srmse
    data_srmse_conv[0][-len(S):,:,0]=Srmse
    data_s_rmse[0][-len(S):,0]=np.transpose(rmse)
    data_s_rmse[0][-len(S):,1:]=S
    data_s_rmse_conv[0][-len(S):,0,0]=np.transpose(rmse)
    data_s_rmse_conv[0][-len(S):,1:,0]=S
  return data,data_conv,data_rmse,data_srmse,data_srmse_conv,data_s_rmse,data_s_rmse_conv

def pred_into_acc(filename):
  data,data_conv,data_rmse,data_srmse,data_srmse_conv,data_s_rmse,data_s_rmse_conv =make_data(filename)
  att_source= np.zeros((1,64))
  z = mse_crs.predict([data_s_rmse_conv,data_s_rmse,att_source])[0]
  y = np.argmax(z)
  return y

def pred_only_text(s):
  rec=featurize_charrnn_utt(s,50)
  att=np.zeros((1,64))
  z = model_fci.predict([rec,att])[0]
  z = np.argmax(z)
  y = int(z)
  if y==0:
    print('부가 정보가 필요합니다...\n')
  elif y==1:
    print('그렇군요. 알겠습니다.\n')
  elif y==2:
    print('궁금하신 점이 있으신 것 같네요.\n')
  elif y==3:
    print('요청 접수되었습니다.\n')
  elif y==4:
    print('진심으로 대답을 원하시는 건 아니죠?\n')
  elif y==5:
    print('일단 알겠습니다.\n')
  else:
    print('목소리가 들렸으면 좋겠어요.\n')
  return z

def pred_audio_text(filename,s):
  rec=featurize_charrnn_utt(s,50)
  att=np.zeros((1,64))
  z = model_fci.predict([rec,att])[0]
  z = np.argmax(z)
  y = int(z)
  if y<6:
    return y
    if y==0:
      print('부가 정보가 필요합니다...\n')
    elif y==1:
      print('그렇군요. 알겠습니다.\n')
    elif y==2:
      print('궁금하신 점이 있으신 것 같네요.\n')
    elif y==3:
      print('요청 접수되었습니다.\n')
    elif y==4:
      print('진심으로 대답을 원하시는 건 아니죠?\n')
    else:
      print('일단 알겠습니다.\n')
  else:
    y_into = pred_into_acc(filename)
    into_att=np.zeros((1,5))
    into_att[0,y_into]=1
    conv=featurize_charcnn_utt(s,50)
    z1 = model_aux.predict([conv,rec,into_att])[0]
    y1 = int(np.argmax(z1))+1
    return y1
    if y==1:
      print('그렇군요. 알겠습니다.\n')
    elif y==2:
      print('궁금하신 점이 있으신 것 같네요.\n')
    elif y==3:
      print('요청 접수되었습니다.\n')
    elif y==4:
      print('진심으로 대답을 원하시는 건 아니죠?\n')
    else:
      print('일단 알겠습니다.\n')

def classify_document(filename):
  target5 = read_data(filename+'.txt')
  f0 = open(filename+'_frag.txt','w')
  f1 = open(filename+'_stat.txt','w')
  f2 = open(filename+'_ques.txt','w')
  f3 = open(filename+'_comm.txt','w')
  f4 = open(filename+'_rheq.txt','w')
  f5 = open(filename+'_rhec.txt','w')
  f6 = open(filename+'_into.txt','w')
  for i in range(len(target5)):
    print(target5[i][0])
    d = pred_only_text(target5[i][0])
    if i%2000 == 0:
        print(i,'over',len(target5))
    if d == 0:
        f0.write(target5[i][0]+'\n')
    if d == 1:
        f1.write(target5[i][0]+'\n')
    if d == 2:
        f2.write(target5[i][0]+'\n')
    if d == 3:
        f3.write(target5[i][0]+'\n')
    if d == 4:
        f4.write(target5[i][0]+'\n')
    elif d==5:
        f5.write(target5[i][0]+'\n')
    else:
        f6.write(target5[i][0]+'\n')
  f0.close()
  f1.close()
  f2.close()
  f3.close()
  f4.close()
  f5.close()
  f6.close()

print('#########################################################\n\n  Intention identification and text classification demo\n\n#########################################################\n\nEnter "i" to identify a sentence\nEnter "c" to classify a corpus\nEnter "bye" to quit\n\n')

from pathlib import Path

def detect():
  print('\nIdentification demo ...\nEnter "c" to activate classification mode\nEnter "bye" to quit\n')
  while 1:
    s = input(' You say: ')
    if s == 'bye':
        sys.exit()
    elif s == 'c':
        classify()
    else:
        pred_only_text(s)

def classify():
  print('\nClassification demo ...\nEnter "i" to activate identification mode\nEnter "bye" to quit\n')
  while 1:
    s = input('Target filename (.txt): ')
    if s == 'bye':
        sys.exit()
    elif s == 'i':
        detect()
    else:
      if Path(s+'.txt').is_file():
        classify_document(s)
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

