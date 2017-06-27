from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras import __version__ as keras_version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


train_raw=pd.read_csv('training_combined.csv')
## read another file containing training data
train2 = pd.read_excel('/home/labcomp/Dropbox/IEDB_DQB10602_training_June16.xlsx', sheetname='Sheet1')
train2.reset_index(drop=True, inplace=True)
train2_nodup=train2.drop_duplicates(['Epitope'])





Positive-Low             366 #2
Positive-Intermediate    287 #2
Negative                 222 #0
Positive-High            142 #1
Positive                  99 #1


train2_nodup.loc[:, ('Assay2')] =0
train2_nodup.loc[train2_nodup['Assay'] == 'Positive-High', ('Assay2')] =1 ## high binders
train2_nodup.loc[train2_nodup['Assay'] == 'Positive-Intermediate', ('Assay2')] =2 ## medium binders
train2_nodup.loc[train2_nodup['Assay'] == 'Positive-Low', ('Assay2')] =2 ## medium binders

train2_nodup.loc[train2_nodup['Assay'] == 'Positive', ('Assay2')] =1 ## high binders
train2_nodup.loc[train2_nodup['Assay'] == 'Negative', ('Assay2')] =0 ## low binders

## recode guo's data
train_raw.Percent.describe()

count    908.000000
mean      69.794800
std       29.049219
min        6.142647
25%       47.262066
50%       75.911056
75%       93.221531
max      145.625939

train_raw.loc[:, ('Assay2')] =0
train_raw.loc[train_raw['Percent'] <= 25, ('Assay2')] = 1 ## high binders
train_raw.loc[(train_raw['Percent'] >= 25) & (train_raw['Percent'] <= 50), ('Assay2')] = 2 ## medium binders
train_raw.shape

X1, X2=train_raw.iloc[:, (2, 5)], train2_nodup.iloc[:, (0, 3)]

X1.columns= ['Sequence', 'Assay2']

X2.columns= ['Sequence', 'Assay2']

X_t=pd.concat([X1, X2], ignore_index=True)
nonstd=X_t.loc[X_t.Sequence.str.contains('\+'), ('Sequence')].tolist()
X_t.loc[X_t.Sequence.str.contains('\+'), ('Sequence')] = [i.split(' ')[0] for i in nonstd]


####################### training ffiles were saved in the directory 
X_t=pd.read_csv('final_training_set_june17.csv')

totalEntries = len(X_t)
maxlen = len(max(X_t.Sequence , key=len))
chars=set("".join(X_t.Sequence))
char_indices = dict((c, i) for i, c in enumerate(chars))
X = np.zeros((totalEntries , maxlen, len(chars) ), dtype=np.bool)
y = np.zeros((totalEntries , 3), dtype=np.int)


y_labels = pd.get_dummies(X_t.Assay2)
for i in xrange(0, len(y_labels)):
	y[i] = y_labels.ix[i]

for i, name in enumerate(X_t.Sequence):
    for t, char in enumerate(name):
        X[i, t, char_indices[char]] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')
#model.fit(X, y, batch_size=16, epochs=20)#, validation_data=(X_test, y_test), shuffle=True)
model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test), shuffle=True)

Y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(accuracy_score(y_test, np.round(Y_pred)))
print(classification_report(y_test, np.round(Y_pred)))

### gives an accuracy of 63% for different classes
## save this model for future use

from keras.models import load_model
model.save('LSTM_model1_june16.h5')
json_string = model.to_json()




print(classification_report(y_test, np.round(Y_pred)))
             #precision    recall  f1-score   support

          #0       0.71      0.75      0.73        87
          #1       0.53      0.58      0.55        33
          #2       0.60      0.54      0.57        83

#avg / total       0.63      0.64      0.63       203


## write out the files for future use
X_t.to_csv('final_training_set_june17.csv')

train_raw.to_csv('final_training_set_june17_GUO.csv')
train2_nodup.to_csv('final_training_set_june17_IEDB.csv')

############## build a function to feed a new input variable for prediction#################


from Bio import Entrez
from Bio import SeqIO
#ADE29095.1
## 
Entrez.email = "ambati@stanford.edu"
handle = Entrez.efetch(db="protein", rettype="fasta", retmode="text", id='ADE29095.1')
seq_record = SeqIO.read(handle, "fasta")


HA_str=str(seq_record.seq)
HA_list =[]
for i in xrange(0, len(HA_str), 1):
	if len( HA_str[i:i+15]) ==15:
		HA_list.append(HA_str[i:i+15])


HAx = np.zeros((len(HA_list), maxlen, len(chars)), dtype=np.bool)
for i, name in enumerate(HA_list):
    for t, char in enumerate(name):
        HAx[i, t, char_indices[char]] = 1




##############other models tested but not accurate ################################################

maxlen = len(max(train_raw.Sequence , key=len))
chars=set("".join(train_raw.Sequence))
char_indices = dict((c, i) for i, c in enumerate(chars))
totalEntries = len(train_raw)
X = np.zeros((totalEntries , maxlen, len(chars) ), dtype=np.bool)
y = np.zeros((totalEntries , 2 ), dtype=np.bool)

for i, name in enumerate(train_raw.Sequence):
    for t, char in enumerate(name):
        X[i, t, char_indices[char]] = 1
    if train_raw.Percent[i] < 25:
    	y[i, 0 ] = 1
    else:
    	y[i, 1] =1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')
#model.fit(X, y, batch_size=16, epochs=20)#, validation_data=(X_test, y_test), shuffle=True)
history1=model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test), shuffle=True)







### other models this gives a 75%
model2 = Sequential()
model2.add(Dense(500, input_shape=(maxlen, len(chars))))
model2.add(Activation('tanh'))
model2.add(Flatten())
model.add(Dropout(0.2))
model2.add(Dense(250))
model2.add(Activation('tanh'))
model2.add(Dense(3))
model.add(Dropout(0.2))
model2.add(Activation('softmax'))
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test))



model3 = Sequential()
model3.add(Dense(500, input_shape=(maxlen, len(chars))))
model3.add(Activation('tanh'))
model3.add(Flatten())
model.add(Dropout(0.4))
#model3.add(Dense(250))
#model3.add(Activation('tanh'))
model3.add(Dense(3))
#model.add(Dropout(0.2))
model3.add(Activation('softmax'))
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model3.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test))




#### try to perform hyperparameter optimization on this dataset


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def create_model(activation='relu'):
	model= Sequential()
	model.add(Dense(500, input_shape=(maxlen, len(chars))))
	model.add(Activation(activation))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(250))
	model.add(Activation(activation))
	model.add(Dropout(0.2))
	model.add(Dense(3))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

model_act = KerasClassifier(build_fn=create_model, epochs=100, batch_size=30, verbose=2)
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model_act, param_grid=param_grid, n_jobs=8)
grid_result = grid.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



model= Sequential()
model.add(Dense(500, input_shape=(maxlen, len(chars))))
model.add(Activation('tanh'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(250))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
