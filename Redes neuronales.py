# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:59:10 2024

@author: josea
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, SimpleRNN, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

carpeta_archivos = r'C:\Users\josea\Documents\Insumos Series de Tiempo'
balance = pd.read_excel(os.path.join(carpeta_archivos,'balance_datos_completos.xlsx'))
vec_fechas = balance.fecha
balance.drop(columns = ['fecha'], inplace = True)
train_df = balance.head(balance.shape[0] - 12)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_df)

balance_escalado = scaler.transform(balance[balance.columns])
balance_escalado = pd.DataFrame(balance_escalado, columns = balance.columns)

def crear_ventana(datos, fechas, sequence_length):
    X, y, dates = [], [], []
    y = datos[sequence_length:]
    dates = fechas[sequence_length:] 
    for i in range(len(datos) - sequence_length):
        X.append(datos[i:i+sequence_length])
    X = np.array(X)

    return X, np.array(y), np.array(dates)

sequence_length = 70
var_indep, var_dep, fechas = crear_ventana(balance_escalado.oblig_publico, vec_fechas, sequence_length)

var_exog = balance[['tasa_desempleo','imae_var_ia','ipc_var_ia','tc_monex_var_ia','tbp']][sequence_length:]
var_exog = var_exog.to_numpy()

X = np.concatenate((var_indep, var_exog), axis = 1)
X = X.reshape((X.shape[0], X.shape[1], 1))  

split_index = len(var_dep) - 24

# Separamos los datos
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = var_dep[:split_index], var_dep[split_index:]
fechas_train, fechas_test= fechas[:split_index], fechas[split_index:]

KERNEL_WIDTH = 3
cnn_model = Sequential(
    [Conv1D(filters=64, kernel_size=(KERNEL_WIDTH,), activation='relu'),
     Dense(units=64, activation='relu'),
     Dense(units=1)
])


#Definimos la CNN
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='tanh'),
    Dropout(0.10),
    Dense(32, activation='tanh'),
    Dense(1)
])

cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.summary()


# Se entrena la CNN
cnn_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3,
                               mode='min')

history2 = cnn_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),
                         callbacks=[early_stopping])


ytest_pred = cnn_model.predict(X_test)

plt.plot(y_test)
plt.plot(ytest_pred)
