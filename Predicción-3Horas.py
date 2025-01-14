import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt

# Obtiene los datos históricos
data = dataset['high'].values.reshape(-1, 1)

# Normaliza los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Divide los datos en conjuntos de entrenamiento y prueba
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Función para crear secuencias de datos
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Define la longitud de la secuencia temporal
sequence_length = 24  # Utilizaremos 24 horas de datos históricos para predecir el precio a 3 horas en el futuro

# Crea secuencias de entrenamiento y prueba
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Crea el modelo LSTM
model = Sequential()
# Agrega una capa de entrada explícita
model.add(Input(shape=(sequence_length, 1)))
# Ahora, puedes agregar la capa LSTM
model.add(LSTM(units=50, activation='relu'))
# Capa de salida
model.add(Dense(units=3))
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrena el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Realiza predicciones
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
predictions = model.predict(X)

# Desnormaliza los datos
predictions = scaler.inverse_transform(predictions)
y = scaler.inverse_transform(y)

# Obtiene la última secuencia de datos históricos
last_sequence = data_scaled[-sequence_length * 3:].reshape(1, -1, 1)

# Predice el precio a 3 horas en el futuro
prediction = model.predict(last_sequence)[:, -3:]
prediction = scaler.inverse_transform(prediction)

# Ajusta el tamaño de la figura
plt.figure(figsize=(16, 6))

# Grafica los datos reales
plt.plot(y, label='Datos reales')

# Grafica las predicciones
plt.plot(predictions, label='Predicciones')
plt.title('Predicción del precio del Bitcoin')
plt.xlabel('Tiempo')
plt.ylabel('Precio')
plt.legend()

# Grafica la pérdida
plt.figure(figsize=(16, 4))
plt.plot(history.history['loss'])
plt.title('Pérdida del modelo')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')

plt.show()
