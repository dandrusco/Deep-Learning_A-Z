# Artificial Neural Network

# Para importar librerias desde Spyder utilizar pip en vez de pip3

# Part 1 - Procesando el Data

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importamos el dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar los datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype=str)
X = X[:, 1:]

# Dividimos el data en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Contruir la RNA (Red neuronales artificales)

# Importamos Keras y librerias adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializamos la RNA, añadimos las capas, compilamos y ajustamos la RNA al entrenamiento
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Evaluar el modelo y calcular prediccion final

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predecir una nueva observacion
"""Utiliza nuestro modelo de RNA para predecir si el cliente con la siguiente informaicon abandonará el banco:
Geografia: France (0.0, 0)
Puntaje de credito: 600
Género: Male (1)
Edad: 40
Tendencia: 3
Balance: 60000
Numerop de productos: 2
Tiene tarjeta de credito: Yes (1)
Cliente activo: Yes (1)
Renta estimada: 50000"""

# A nuestra nueva prediccion deberemos transformarla ya que antes realizamos un escalado (sc)
# Crearemos una matriz bidimencional con Numpy para crear una sola fila, para introducir los datos de la nueva observacion
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Elaboraremos una matriz de confusion para visualizar mejor estos datos
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)