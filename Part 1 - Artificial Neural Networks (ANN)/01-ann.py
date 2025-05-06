# Artificial Neural Network

# Instalar Keras desde Anaconda
# conda install -c conda-forge keras

# Para importar librerias desde Spyder utilizar pip en vez de pip3

# Part 1 - Procesando el Data

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importamos el dataset
dataset = pd.read_csv('Churn_Modelling.csv')

#Seleccionamos las columnas de las variables independientes:
#Puntuacion de credito, geografia, genero, edad, tipo de cliente, balance, numero de productos, 
    #tarjeta de credito, miembro activo, sueldo estimado
X = dataset.iloc[:, 3:13].values

#Seleccionamos la columna de la variable dependiente (la que queremos predecir) si el cliente esta o se fue 
y = dataset.iloc[:, 13].values

# Codificar los datos categoricos como el pais (Francia, españa o alemania) y el genero (Hombre o mujer) 
    # para transformarlos en numeros
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype=str)

# Con esto logreamo que hemos creado 2 columnas para los paises, la primera es si es frances
# La segunda si es Aleman, y la tercera su es esopañol.
# Eliminamos la primera columna, ya que si la fila 2 y la 3 es igual a 0, entonces sabremos que es frances
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

# Inicializamos la RNA, para clasificar de la clase Sequential()
classifier = Sequential()

# Añadiremos las capas de entradas y la primera capa oculta, para ello tomaremos classifier y le añadiremos:
# la capa de Dense,en Units seran los nodos de la capa oculta, como consejo si tenemos 11 variables le pondremos la mitad, en este caso 6
# En kernel_initializer es como inicializaremos esos pesos, puede ser como constante, uniforme, etc.
# en Activation es la funcion de activacion para la capa oculta, elijiremos RELU (rectificacion lineal unitario)
# Y por ultimo el input_dim, que son los nodos de entrada, que son nuestras 11 variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Añadiremos una segunda capa oculta, esta segunda capa ya sabe los parametros de entrada, asi que no los ponemos
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Añadiremos la capa de salida, por lo tanto necesitamos solo 1 nodo en la capa de salida
# Pero la capa de activacion la pondremos como sigmoid para que tenga como valor 0 o 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Debemos compilar nuestra Red Neuronal Artificial
# optimizer: se encarga de optimizar, el de serie es adam
# loss: corresponde a la funcion de perdida, quien minimiza el error, utilizamos binary_crossentropy , para transformar las categorias en numeros
# metrics: es la precicion, utilizamos accuracy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Ajustamos la RNA al conjunto de entrenamiento
# Primero agregamos el conjunto a entrenar: X_train y el segundo es el que quiero prececir: y_train
# batch_size: Corresponde al numero de bloques (procesa 10 elemento y luego corrige los pesos)
# epochs: son las repeticiones (iteraciones)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Podemos ver que en la ultima iteracion:
#Epoch 100/100
#800/800 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8298 - loss: 0.4107 

# Que significa? que cada 100 datos que le suministremos, un 83% realizara una buena prediccion 


# Part 3 - Evaluar el modelo y calcular prediccion final

# Prediccion de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)
# Seleccionaremos el y_pred y podemos ver que el cliente 5 y 9 tiene mas probabilidad de avandonar el banco
# Podemos filtrar si solo queremos ver los clientes a un mayor al 50% en la prediccion
y_pred = (y_pred > 0.5)
# Vemos que ahora lo transformamos en Booleano, para ver si avandona o no 

# Elaboraremos una matriz de confusion para visualizar mejor estos datos
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Ahora podemos seleccionar CM de las variables explorer (derecha supertior de Spyder)
# de las 2.000 observaciones 1.539 + 143 son predicciones correctas, mientras que 262 + 56 son incorrecta (datos cruzados)
# Sumamos 1.539 + 143 = 1.682 y los dividimos en 2.000, nos da 84.1%