## Mega Caso de Estudio 

# Parte 1 - Identificar los fraudes potenciales con un SOM

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
#dataset = pd.read_csv("Credit_Card_Applications.csv")
dataset = pd.read_csv("Credit_Card_Applications.csv", dtype={'CustomerID': int})
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Entrenar el SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizar los resultados
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5,
         markers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None', 
         markersize = 10, markeredgewidth = 2)
show()

# Encontrar los fraudes
mappings = som.win_map(X)
frauds = np.concatenate( (mappings[(7,1)], mappings[(1,6)]), axis = 0 )
frauds = sc.inverse_transform(frauds)


# Parte 2 - Trasladar el modelo de Deep Learning de no supervisado a supervisado

# Crear la matriz de características, para ello crearemos una variable llamada customers(clientes)
# Omiteremos la primera columna del identificador del cliente y class ya que la idea es predecirla nosotros mismo
customers = dataset.iloc[:, 1:-1].values


# Crear la variable dependiente llamada is_fraud (es fraude?)
# crearemos un vector de ceros de la misma longitud del dataset
is_fraud = np.zeros(len(dataset))

# Vamos a recorrer uno todas las observaciones en un rango del dataset
for i in range(len(dataset)):
    # vamos a comprobar si el individuo en la fila "i" tiene un identificador  que este en la lista de los fraudes
    if dataset.iloc[i, 0] in frauds:
        # De ser así le cambiaremos del 0 al 1
        is_fraud[i] = 1

# Escalado de variables en customers
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)


# Parte 2 - Construir la RNA

# Importamos Keras y librerias adicionales
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta, input_dim con las 14 columnas
classifier.add(Dense(units = 2, kernel_initializer = "uniform",  activation = "relu", input_dim = 14))


# Añadir la capa de salida y como es una provabilidad, la activaremos como  sigmoid
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA sera el de adam
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
# En la entrada le pasamos los clientes (customers) y la prediccion es is_fraud
# el entrenamiento de bloque sera de 1 en 1 y las epocas haremos 2 pasadas
classifier.fit(customers, is_fraud,  batch_size = 1, epochs = 2)


# Predicción de los resultados de fraude
y_pred  = classifier.predict(customers)
# Vamos a concatenar la columna de los identificadores de los clientes con la prediccion, con axis 1 para concatenar por columna
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis = 1)
# Vamos a ordenar los datos del mas probable al menos probable en cuanto al posible fraude
y_pred = y_pred[y_pred[:, 1].argsort()[::-1]]

# Guardaremos este resultado en un csv
result_df = pd.DataFrame(y_pred, columns=["CustomerID", "Prediction"])
result_df["CustomerID"] = result_df["CustomerID"].astype(int)
result_df.to_csv("predicciones.csv", index=False)