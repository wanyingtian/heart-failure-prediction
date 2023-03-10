import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

## Loading Data
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(data.info())
print(Counter(data['DEATH_EVENT']))
y = data['DEATH_EVENT']
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

## Data Pre-processing
x = pd.get_dummies(x)
# split datasets to train and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=1)
# standardize numerical values
ct = ColumnTransformer([("numeric",StandardScaler(),['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])
# apply standardization to training and testing features
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)

## Prepare labels for classification
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))
# transform labels into a binary vector
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

## Design the model
model = Sequential()
#input layer
model.add(InputLayer(input_shape = (X_train.shape[1],)))
# hidden layer
hidden_layer = Dense(12,activation='relu')
model.add(hidden_layer)
# output layer
model.add(Dense(2,activation='softmax'))
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Train and evaluate
history = model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2)
val_loss, val_acc = model.evaluate(X_test, Y_test, verbose = 0)
print("Loss", val_loss, "Accuracy:", val_acc)

y_estimate = model.predict(X_test, verbose = 0)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)
print(classification_report(y_true, y_estimate))

fig = plt.figure()
# plot loss and accuracy over each epoch

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')


# Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
plt.show()