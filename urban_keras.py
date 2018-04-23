import librosa
import librosa.display
import csv
import os
import subprocess
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


#Converts mp3 files to wav files 
def mp3towav(dir):
    for filename in os.listdir(dir):
        if(dir.split('\\')[-2]=='Train'):
            subprocess.call(['ffmpeg','-i',dir+filename,dir[:-6]+'Train16bit\\'+filename[:-4]+'.wav'])
        else:
            subprocess.call(['ffmpeg','-i',dir+filename,dir[:-5]+'Test16bit\\'+filename[:-4]+'.wav'])
       
    
mp3towav('C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\urban_sounds\\train\\Train\\')  
mp3towav('C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\urban_sounds\\test\\Test\\')  


np.random.seed(7)
tr_dataset=pd.read_csv("C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\urban_sounds\\train\\train.csv")
ts_dataset=pd.read_csv("C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\urban_sounds\\test\\test.csv")


#Extracts label of audio file and it's MFCCS feature
def loadFileAndFeature(dir):
    mfccs=[]
    label=[]
    reader=csv.reader(open(dir+"\\"+dir.split('\\')[-1].lower()+".csv","r"),delimiter=',')
    next(reader)
    if(dir.split('\\')[-1]=='train'):
        for row in reader:  
            y, sr = librosa.load(dir+'\\T'+dir.split('\\')[-1][1:]+'16bit\\'+row[0]+'.wav')
            mfccs.append(np.mean(librosa.feature.mfcc(y,sr,n_mfcc=40).T,axis=0))
            label.append(row[1])
    else:
        for row in reader:  
            y, sr = librosa.load(dir+'\\T'+dir.split('\\')[-1][1:]+'16bit\\'+row[0]+'.wav')
            mfccs.append(np.mean(librosa.feature.mfcc(y,sr,n_mfcc=40).T,axis=0))
    return label,mfccs


#def loadFileAndFeature1(row):
#    y, sr = librosa.load(dir+'\\T'+dir.split('\\')[-1][1:]+'16bit\\'+str(row.ID)+'.wav')
#    mfccs=(np.mean(librosa.feature.mfcc(y,sr,n_mfcc=40).T,axis=0))
#    label=(row.Class)
#    return [label,mfccs]
#tr=tr_dataset.apply(loadFileAndFeature,axis=1)
#tr.columns('label','feature')

tr_label, tr_features=loadFileAndFeature("C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\urban_sounds\\train")
tr_features=np.array(tr_features)
tr_label=np.array(tr_label)

#ts=ts_dataset.apply(loadFileAndFeature,axis=1)
#ts.columns=['label','feature']

ts_label, ts_features=loadFileAndFeature("C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\urban_sounds\\test")
ts_features=np.array(ts_features)
ts_label=np.array(ts_label)

lb=LabelEncoder()
encoded_label=np_utils.to_categorical(lb.fit_transform(tr_label))
distinct_labels=encoded_label.shape[1]

del model

model=Sequential()
model.add(Dense(100,input_shape=(40,),activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(distinct_labels,activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.fit(tr_features,encoded_label,epochs=30,validation_split=0.2)

predicted_class = model.predict_classes(ts_features)
output=lb.inverse_transform(predicted_class)
print(output)

x=output.tolist()
x=pd.DataFrame(x)
x.columns=['Class']

df=pd.concat([x,ts_dataset],axis=1)

df.to_csv("C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\urban_sounds\\test\\output.csv",sep=',',index=False)


#Plotting graphs for data and RFFT
y, sr = librosa.load('C:\\Users\\I333690\\Desktop\\Fellowship\\Audio_files\\Dataset\\coffee_machine\\WAV16\\0.wav')
plt.figure(figsize=(12,4))
plt.suptitle('y',x=0.5,y=0.8,fontsize=20)
plt.xlabel('Samples of time')
plt.ylabel('Amplitude')
plt.plot(y)

data_rfft=np.fft.rfft(y)
plt.figure(figsize=(12,4))
plt.suptitle('RFFT frequencies',x=0.5,y=0.8,fontsize=20)
plt.xlabel('Samples of time')
plt.ylabel('Frequency')
plt.plot(data_rfft)
