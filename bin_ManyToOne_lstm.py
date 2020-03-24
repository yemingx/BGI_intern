#input example:
#time python binary_multi_to_one_lstm.py ../input/NB02_NB03_feature.xls.PLUS.7nt 1000 1000 0.0 0.0 softmax 4 adam

import numpy as np # linear algebra
import pandas as pd
import os
import sys
import re
import seaborn as sns
import matplotlib.pyplot as plt

from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Flatten, Dense, Bidirectional, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras import optimizers

with open(sys.argv[1]) as f:
	line = f.read().splitlines()

matrix_list = []
for a in range(0,len(line)):
    item = line[a].split(';')
    y = [float(item[0].split('\t')[0]),float(0),float(0)]
    read_list = [y]
    #print(y, item[0])
    for b in range(1,len(item)):
        feature = item[b].split(',')
        feature_array = [float(i) for i in feature]
        read_list.append(feature_array)
    #print(read_list)
    read_array = np.asarray(read_list)
    matrix_list.append(read_array)
matrix_array = np.asarray(matrix_list)
matrix_array.shape
matrix_array

##random 20% rows for train_val, 20% rows for test
N = matrix_array.shape[0]
np.random.shuffle(matrix_array)

train_train = matrix_array[:int(N*0.6)]
train_val = matrix_array[int(N*0.6):int(N*0.8)]
test = matrix_array[int(N*0.8):]
print('train_train: ' + str(train_train.shape) + '; train_val: '+ str(train_val.shape) + 
     '; test: ' + str(test.shape) + '\n')


x_train = train_train[:,1:8]
y_train = train_train[:,0]
y_train = np.delete(y_train, [1,2],axis=1)
print('x_train.shape: '+ str(x_train.shape) +'; y_train.shape: '+ str(y_train.shape))

x_val = train_val[:,1:8]
y_val = train_val[:,0]
y_val = np.delete(y_val, [1,2],axis=1)
print('x_val.shape: '+ str(x_val.shape) +'; y_val.shape: '+ str(y_val.shape))

x_test = test[:,1:8]
y_test = test[:,0]
y_test = np.delete(y_test, [1,2],axis=1)
print('x_test.shape: '+ str(x_test.shape) +'; y_test.shape: '+ str(y_test.shape))

batch_size_input, epochs_input, dropout_input = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
rc_dropout_input = float(sys.argv[5])
activation_input = sys.argv[6]
LSTM_unit_input = int(sys.argv[7])
optimizer = sys.argv[8]
lr = float(sys.argv[9])
if optimizer == 'adam': 
	optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

print('input_sample: ' + sys.argv[1] + '\nbatch_size: ' + sys.argv[2] + ', epochs: ' + sys.argv[3] 
    + ', dropout: ' + sys.argv[4] + ', rc_dropout: ' + sys.argv[5] + '\nactivation: ' + sys.argv[6] 
    + ', LSTM_unit: ' + sys.argv[7] + ', optimizer: ' + sys.argv[8] + ', learning rate: ' + sys.argv[9] + '\n')

model = Sequential()
model.add(BatchNormalization(input_shape=(7, 3)))
model.add(Bidirectional(LSTM(LSTM_unit_input, dropout=dropout_input, 
    recurrent_dropout=rc_dropout_input, activation=activation_input, return_sequences=True)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary)

model.fit(x_train, y_train, batch_size=batch_size_input, epochs=epochs_input, validation_data=(x_val, y_val))

# Get accuracy of model on validation data. It's not AUC but it's something at least!
score, acc = model.evaluate(x_val, y_val, batch_size=1000)
print('Eval accuracy: {:.4f} %'.format(float(acc)*100))

y_predict = model.predict(x_test, batch_size=1000)
print("test accuracy: {:.4f} %".format(100 - np.mean(np.abs(y_predict - y_test)) * 100))
print('input_sample: ' + sys.argv[1] + '\nbatch_size: ' + sys.argv[2] + ', epochs: ' + sys.argv[3] 
    + ', dropout: ' + sys.argv[4] + ', rc_dropout: ' + sys.argv[5] + '\nactivation: ' + sys.argv[6] 
    + ', LSTM_unit: ' + sys.argv[7] + ', optimizer: ' + sys.argv[8] + ', learning rate: ' + sys.argv[9] + '\n')
