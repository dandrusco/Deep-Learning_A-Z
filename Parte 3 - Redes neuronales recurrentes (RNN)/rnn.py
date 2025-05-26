# Redes Neuronales Recurrentes (RNN)



# Parte 1 - Preprocedado de los datos

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importamos el dataset de entrenamiento con los datos de las acciones de google entre el 2012 y el 2016
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# Nos quedaremos solo con la columnas de apertura de la bolsa del dataset (Open)
training_set = dataset_train.iloc[:, 1:2].values

# Escalado de los datos, utilizaremos el escalado MinMaxScaler de procesado de  sklearn
# Donde el valor mas pequeño se transforma en cero y el maximo en 1 
sc = MinMaxScaler(feature_range = (0, 1))
# Transformaremos nuestro conjunto de entrenamiento en un conjunto de entrenamiento escalado
training_set_scaled = sc.fit_transform(training_set)

# Crearemos una estructura de datos con 60 timesteps y 1 salida, lo que equivale que la RNN mirara 60 dias hacia atras
# 60 dias equivale a 1 trimestre, teniendo ese dato tomar el dato 0 al 60 y tratara de predecir el dia 61 (dia siguiente)
# Luego tomara del dia 1 al 61 para predecir el dia 62 y asi sucesivamente 
# Crearemos 2 listas, una para el conjunto de entrenamiento (la entrada) y la otra para el conjunto de salida
X_train = []
y_train = []

# Comenzaremos a rellenar los datos con un bucle for, la primera iteracion partira en el 60 y el ultimo sera 1258
for i in range(60, 1258):
    # El X_train se debera llevar bloques de 60, por lo tanto nos vamos 60 dias hacia atras i-60 y llegara hasta el 60
    # Como los conjuntos de datos son bidimencionales debemos parasarle un 0 
    X_train.append(training_set_scaled[i-60:i, 0])
    # En y_train apendizaremos el dia de la prediccion actual 61, 62, 63... (tambien le debemos pasar un 0 al final)
    y_train.append(training_set_scaled[i, 0])
# Vamos a pescar ambos conjuntos para transformarlas en matrices de numpy
X_train, y_train = np.array(X_train), np.array(y_train)

# Redimension de los datos
# Con reshape podemos agregar una nueva dimencion al conjunto de datos
# El primer argumento es lo que queremos redimencionar, el segundo parametro es una tupla
# En la tupla le pasamos la nueva estructura, la primera es [0] para obtener el numero de filas
# la segunda es [1] para el numero de columnas y 1 como variable predictoria 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Al ejecurar ahora el X_train se transformo a 1198, 60, 1
# Al abrir el X_train en Variable Explirer vemos una solo columna, abajo en Axis podemos cambiar a las otras 0, 1 o 2


# Parte 2 - Construir la RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Parte 3 - Ajustar las predicciones y visualizar los resultados

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
