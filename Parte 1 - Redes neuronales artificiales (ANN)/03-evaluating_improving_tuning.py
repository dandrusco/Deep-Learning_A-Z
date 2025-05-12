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

# No utilizaremos la parte 2 de Contruir la RNA, ni la Part 3 de Evaluar el modelo y calcular prediccion final

# Part 4 - Evaluar, mejorar y ajustar la RNA

# Evaluar la RNA
from scikeras.wrappers import KerasClassifier #Si da error instalar pip install scikeras
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense

# Crearemos una funcion para la clasificacion 
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#Creamos el clasificador de KerasClassifier donde el build_fn, llamara a nuestra funcion
classifier = KerasClassifier(model=build_classifier, batch_size=10, epochs=100, verbose=0)

# Definiremos la precicion accuracies, invocando al cross_val_score (validacion cruzada) 
# Como parametro estimador, le pasamos el clasificador, X para el datos X_train y el Y para las preciciones pero de y_train (no utilizamos el test)
# con cv correesponde al numero de validaciones cruzadas, y el n_jobs es el numero de tareas simultaneo (CPU), un -1 seleccionamos todas
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1, verbose=1)

mean = accuracies.mean()
variance = accuracies.std()
print(mean)
print(variance)

# Mejora de la ANN
# Regularización de la pérdida de datos para reducir el sobreajuste, si es necesario

# Ajustar la ANN
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# Nuestra funcion ahora recibira un parametro para la optimizacion.
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Los parametros del KerasClassifier ahora va solo con el build_fn que pasa a llamarse Model, sin batch_size, epochs ni verbose
classifier = KerasClassifier(model=build_classifier)

# Ahora estos parametros los meteremos en un diccionario para que combiene los valores 
# en batch_size pondremos potencias de 2: 25 y 32
# La iteraciones pondremos 100 y 500
# Y en la opcimizacion podremos dos 'adam' y 'rmsprop'
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'model__optimizer': ['adam', 'rmsprop']}

# Ahora definiremos el objeto GridSearchCV, el estimator sera classifier y param_grid corresponde a los parameters
# La metrica en scoring sera la accuracy y por ultimo la cros validation sera de 10)
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

# Ahora es el ultimo de ajustar los datos de X_train para lograr predecir la y_train
grid_search = grid_search.fit(X_train, y_train)

# Nos quedaremos con los mejores marametros que nos devuelva el grid_search
best_parameters = grid_search.best_params_

# Y por otro lado nos quedaremos con la mejor precicion de la validacion cruzada
best_accuracy = grid_search.best_score_