# Self Organizing Map

# Importamos las librerias 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importaremos el dataset del formulario para pedir tarjetas de creditos al banco
dataset = pd.read_csv('Credit_Card_Applications.csv')
# La idea es poder detectar que usuarios han mentido en sus datos para obtener estas tarjetas
# Nos quedaremos con todas las columnas menos la ultima llamada Class ya que esa es la prediccion
X = dataset.iloc[:, :-1].values
# Para la prediccion nos quedamos con la ultima, Class nos indica si fue o no aprobada. 
y = dataset.iloc[:, -1].values

# Escalado de caracteristicas
from sklearn.preprocessing import MinMaxScaler
# Escalaremos entre 0 y 1
sc = MinMaxScaler(feature_range = (0, 1))
# Ajustamos los datos de X para que todas las columnas queden entre 0 y 1
X = sc.fit_transform(X)


# Entrenar el SOM
# En la carpeta esta el archivo minisom.py Es una libreria que ya esta lista para poder utilizar los MapasAutoOrganizados
# Imoportamos la libreria
from minisom import MiniSom

# Comenzaremos a buscar patrones con MiniSom, en X e Y son las dimenciones, en nuestro caso sera de 10x10
# En cuanto al input_len corresponde a la longitud de la entrada, nosostros tenemos 15 columnas en X
# Sigma corresponde al radio inicial para establecer los vecinos, por defecto es 1.0
# learning_rate se encarga de ir adaptando los pesos, por defecto es 0.5
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Ahora devemos inicializar los pesos cercano a 0
som.random_weights_init(X)
# Comenzaremos a entrar de forma aleatoria, pasandole el conjunto de dato y el numero de iteraciones
som.train_random(data = X, num_iteration = 100)


# Visualizar el resultado
# Importamos las librerias necesarias para pintar el mapa
from pylab import bone, pcolor, colorbar, plot, show
# Ejecutamos la funcion bone y vemos que se ejecuta una nueva ventana al estilo TKinter en Python
bone()
# Generaremos un rango de colores, apartir del mapa de las distancias del propio som
# con T giramos la matriz (por defecto viene en filas y las necesitaremos en columnas)
pcolor_result = pcolor(som.distance_map().T)
# Fabricampos una barra de color
colorbar(pcolor_result)
# Estableceremos los marcadores, el marcador del circulo quedara con la letra O y el marcador del cuadrado con la S
markers = ['o', 's']
# Lo mismo para los colores, los circulos de color rojo y los cuadrado de color verde
colors = ['r', 'g']
# Creamos un bucle, la I es para la posicion y la X para los valores especificos, en enumerate de los datos X
for i, x in enumerate(X):
    # Obtenemos el nodo ganador de X
    w = som.winner(x)
    # Creamos el plot para pintar el circulo o cuadrado con sus colores espectivamente
    # W nos devolvera una tupla con dos valores, [0] nos devuelve la X y [1] nos devuelve la Y, le sumaremos 0.5 para centrarlo
    # markers obtenemos la prediccion de la Y en la posicion I, devolvera 0 y 1 para ir pintandolo
    # Ahora lo mismo para colors, pero se lo pasaremos a la variable markeredgecolor para que nos pinte el borde
    # Para asegurarnos que no nos pinte el interior utilizamos markerfacecolor = 'None',
    # markersize es el tamaño del circulo y el cuadrado 
    # markeredgewidth lo hacemos mas grueso
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)

# Para indicar que terminamos el plot y lo queremos visualizar, utilizamos show()
show()

# Encontrar los fraudes
# Crearemos un mapeo de todos los nodos ganadores
mappings = som.win_map(X)
# Crearemos una variable para los fraudes, donde 
# El mappings deberemos poner una tupla del cuadrado mas blanco que nos dejo nuestro grafico, en este caso en la posicion 3,1
# El cuadrado que esta en la posicion 0,0 del tablero sera el 0, por lo tanto el de al lado sera el 1, lo mismo hacia arriba
# El segundo mappings sera el segundo cuadrado mas claro, poniendole su cordenada, en este caso la 8.7 
# Para juntar los invividuos que caen tanto en el primero como el segundo, deberemos contatenarlo con numpy
# y el axis sera en 0 ya que la union sera en fila
frauds = np.concatenate((mappings[(3,1)], mappings[(8,7)]), axis = 0)
# Esto nos devolvio los posibles individuos fraudulentos, en mi caso 55 
# Como los valores estan escalado, deberemos crear un escalado inverso
frauds = sc.inverse_transform(frauds)
# Ahora ya se podria revisar uno a uno para ver si estan haciendo fraudes para obtener las tarjetas de creditos