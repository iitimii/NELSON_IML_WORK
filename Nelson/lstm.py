import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


df = pd.read_csv('train2dof/data_sample_2022-02-21-07-46-41/marker.csv')

df.to_numpy()

data = np.reshape(df, (-1, 10, 16))
labels = int
tx = 10
n_a = 16
n_x = 16

reshape = Reshape((1, n_x))
lstm = LSTM(n_a, return_state=True)
D = Dense(1, activation='sigmoid')

X = Input(shape=(tx,n_a))
a0 = Input(shape=(n_a,), name='a0')
c0 = Input(shape=(n_a,), name='c0')
a = a0
c = c0

outputs = []

for t in range(tx):
    x = X[:,t,:]
    x = reshape(x)
    a, _, c = lstm(x, [a, c])
out = D(a)
outputs.append(out)

model = Model(inputs=[X, a0, c0], outputs=outputs)

print(model.summary())


m = int
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

model.compile(optimizer=Adam(lr=0.01, decay=0.01), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit([data, a0, c0], labels, epochs=100)