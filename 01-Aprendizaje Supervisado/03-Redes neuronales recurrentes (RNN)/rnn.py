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

# Importamos las librerias de Keras
from keras.models import Sequential
# De las capas de Keras importaremos la densa
from keras.layers import Dense
# De las capas de Keras importaremos la LSTM para apilar 
from keras.layers import LSTM
# De las capas de Keras importaremos la Dropout para prevenir el overfiting
from keras.layers import Dropout

# Crearemos una regrecion de secuencias de capas
regressor = Sequential()

# Añadiremos la primera capara de LSTM y la regulariacion por Dropout
# A la regrecion le añadiremos una capa de LSTM, como paramatro le pasaremos cuantas neuronas queremos meter (units)
# EL segundo parametro es la secuencia de retorno, si queremos unas varias capas apiladas, debemos ponerlo en True
# El input_shape corresponde al tamaño y dimenciones de los datos 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Ahora le añadiremos un Dropout para desactivar algunas neurnas del paso anterior para prevenir el overffiting 20%
regressor.add(Dropout(0.2))

# Añadiremos la segunda capa de LSTM y la regulariacion por Dropout
# Sera la misma capa de LSTM y Dropout pero ya no necesitamos los datos de entrada
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Añadiremos la tercera capa de LSTM y la regulariacion por Dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Añadiremos la cuarta capa de LSTM y la regulariacion por Dropout
# En la ultima capa ya no necesitamos return_sequences por que esta sera una capa densa
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Añadiremos la capa de salida a la regresion, sera una capa densa con 1 neurona de salida
regressor.add(Dense(units = 1))

# Compilamos la RNN, el optimizador sera adam que se utiliza para las redes neuronales recurrentes
# El segundo parametro es de las perdidas a optimizar, utilizaremos mean_squared_error 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Ajustaremos la RNN a nuestro conjunto de entrenamiento
# con fit ajustamos los datos, como parametro le pasamosla capa de entrada, prediccion de salida, epochs y batch_size
# le pasaremos 100 iteraciones (epocas) y con el tamaño del bloque (batch_size) seran de 32 
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
# Al compilar esta linea, la primera iteracion nos da un error (loss) de un 4% 
#  38/38 [==============================] - 6s 50ms/step - loss: 0.0494
# A medida que avanza y nos acercamos al 100, vemos como el error va disminuyendo, hasta un 0.15%


# Parte 3 - Ajustar las predicciones y visualizar los resultados

# Obter el valor real de las acciones de Enero de 2017, para ello cargaremos el dataset Google_Stock_Price_Test
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
# Nos quedaremos solo con la columnas de apertura de la bolsa del dataset (Open)
real_stock_price = dataset_test.iloc[:, 1:2].values

# Predecir las acciones de Enero de 2017 con la RNN
# Juntaremos el dataset de entrenamiento y el de pruebas (uno bajo del otro, axis = 0 )
# solo nos quedaremos con la columna Open
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# Necesitamos quedarnos con el utimo dia del dataset_train, por lo tanto tomaremos la longitud de todos los datos
# Y se la restaremos a la longitud de los datos de entrenamiento (1278 - 20), asi ontenemos el ultimo dia del 2016
# Obtendremos los datos de los 60 dias anteriores (-60), con los : obtenemos los datos de la fecha hacia adelante
# Por lo tanto obtenemos en esta linea los ultimos 60 dias del 2016 mas los 20 dias del 2017, total 80 dias
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# Hasta aqui, hemos creado un vector de fila con 80 datos, unos tras otros, pero nosotros necesitamos en columnas
# Con reshape pasamos de filas a columnas con (-1,1)
inputs = inputs.reshape(-1,1)
# Ahora debemos escalar los datos si o si para que pueda trabajar en esta red neuronal recurrente
# El Escalado no es necesario hacerlo con fit por que el minimo y maximo ya lo tenemos 
inputs = sc.transform(inputs)

# Crearemos 1 listas para poder hacer las predicciones
X_test = []

# Comenzaremos a rellenar los datos con un bucle for, la primera iteracion partira en el 60 y el ultimo sera de 80
# Para poder predecir los 20 dias del 2017
for i in range(60, 80):
    # El X_test se llevara bloques de 60 del imputs, por lo tanto nos vamos 60 dias hacia atras
    # Como los conjuntos de datos son bidimencionales debemos parasarle un 0 
    X_test.append(inputs[i-60:i, 0])

# Vamos a pescar el conjuntos de X_test para transformarlas en matrices de numpy
X_test = np.array(X_test)
# Toca redimencionar los datos. Con reshape agregamos la nueva dimencion
# El primer argumento es lo que queremos redimencionar, el segundo parametro es una tupla
# En la tupla le pasamos la nueva estructura, la primera es [0] para obtener el numero de filas
# la segunda es [1] para el numero de columnas y 1 como variable predictoria 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Ahora es el turno predecir los 20 dias de enero de 2017, para ello el regressor hara una prediccion del X_test
predicted_stock_price = regressor.predict(X_test)
# Como todos los datos los escalamos entre 0 y 1, deberemos revertir la transformacion para ver el dato de la prediccion
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizar los resultados
# Pintamos los valores de real_stock_price (datos reales), de color rojo con la etiqueta 
plt.plot(real_stock_price, color = 'red', label = 'Precio Real de la Accion de Google')
# Pintaremos ahora la prediccion de color azul 
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio Prediccion de la Accion de Google')
# Le agregamos un titulo 
plt.title('Prediccion con una RNR del valor de las acciones de Google')
# Creamos las etiquetas del eje X e Y
plt.xlabel('Tiempo')
plt.ylabel('Precio de la accion de Google')
# Habilitamos la leyenda
plt.legend()
# Arrancamos el grafico
plt.show()
