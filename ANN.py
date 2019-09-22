
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

x_train=x_train.reshape(x_train.shape[0],-1)/255.0
x_test=x_test.reshape(x_test.shape[0],-1)/255.0
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential()
model.add(Dense(10,input_dim=784,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,validation_split=0.1)

test_acc=model.evaluate(x_test,y_test)

model2=Sequential()
model2.add(Dense(50,input_dim=784,activation='relu'))
model2.add(Dense(10,activation='softmax'))
model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.fit(x_train,y_train,epochs=10,validation_split=0.1)

model2.evaluate(x_test,y_test)

model3=Sequential()
model3.add(Dense(50,input_dim=784,activation='relu'))
model3.add(Dense(50,activation='relu'))
model3.add(Dense(10,activation='softmax'))
model3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model3.fit(x_train,y_train,epochs=20,validation_split=0.1)

from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten
import numpy as np
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
x_train=x_train[:,:,:,np.newaxis]/255.0
x_test=x_test[:,:,:,np.newaxis]/255.0
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
y_test.shape

model4=Sequential()
model4.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu',input_shape=(28,28,1)))
model4.add(MaxPool2D(pool_size=2))
model4.add(Flatten())
model4.add(Dense(10,activation='softmax'))
model4.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model4.summary()
model4.fit(x_train,y_train,epochs=10,validation_split=0.1)

from tensorflow.keras.layers import Dropout
model5=Sequential()
model5.add(Conv2D(filters=64,kernel_size=2,padding='same',activation='relu',input_shape=(28,28,1)))
model5.add(Conv2D(64,kernel_size=2,activation='relu'))
model5.add(MaxPool2D(pool_size=2))
model5.add(Dropout(0.25))
model5.add(Flatten())
model5.add(Dense(10,activation='softmax'))
model5.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model5.fit(x_train,y_train,epochs=10,validation_split=0.1)

