#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from IPython.core.display import display, HTML
# stop annoying tensorflow warning messages
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

sdir=r'instaImages' 
classlist=os.listdir(sdir)    
filepaths=[]
labels=[]    
for klass in classlist:
    classpath=os.path.join(sdir,klass)        
    flist=os.listdir(classpath)        
    for f in flist:
        fpath=os.path.join(classpath,f)        
        filepaths.append(fpath)
        labels.append(klass) 
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
df=pd.concat([Fseries, Lseries], axis=1 )       
trsplit=.9
vsplit=.05
dsplit=vsplit/(1-trsplit)
strat=df['labels']
train_df, dummy_df=train_test_split(df, train_size=trsplit, shuffle=True, random_state=123,stratify=strat)
strat=dummy_df['labels']
valid_df, test_df=train_test_split(dummy_df, train_size=dsplit, shuffle=True, random_state=123,stratify=strat)
# print('train_df length: ', len(train_df), '  test_df length: ',len(test_df), '  valid_df length: ', len(valid_df))
balance=list(train_df['labels'].value_counts())
# print (balance)

height=500
width=250
channels=3
batch_size=30
img_shape=(height, width, channels)
img_size=(height, width)
length=len(test_df)
test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
test_steps=int(length/test_batch_size)
# print ( 'test batch size: ' ,test_batch_size, '  test steps: ', test_steps)
def scalar(img):    
    return img  # EfficientNet expects pixelsin range 0 to 255 so no scaling is required
# trgen=ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
# tvgen=ImageDataGenerator(preprocessing_function=scalar)
trgen=ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input, horizontal_flip=True)
tvgen=ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input)
train_gen=trgen.flow_from_dataframe( train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen=tvgen.flow_from_dataframe( test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=False, batch_size=test_batch_size)
valid_gen=tvgen.flow_from_dataframe( valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                    color_mode='rgb', shuffle=True, batch_size=batch_size)
classes=list(train_gen.class_indices.keys())
class_count=len(classes)
train_steps=np.ceil(len(train_gen.labels)/batch_size)


model_name='EfficientNetB2'
base_model=tf.keras.applications.EfficientNetB2(weights=None,include_top=False,input_shape=img_shape, pooling='max')
x=base_model.output       
output=Dense(class_count, activation='softmax')(x)
model=Model(inputs=base_model.input, outputs=output)
model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy']) 

history=model.fit(x=train_gen,  epochs=1, verbose=0, validation_data=valid_gen,
               validation_steps=None,  shuffle=False,  initial_epoch=0)

print("Validation accuracy:",*["%.8f"%(x) for x in
     history.history['val_sparse_categorical_accuracy']])