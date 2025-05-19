# Redes Neuronales Convolucionales

# Parte 1 - Construir el modelo de CNN

# Importar librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializamos la CNN (Redes Neuronales Convolucionales)
classifier = Sequential()

# Paso 1 - Convolucion
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Añadiremos una segunda capa de convolucion y maxPooling para mejorar la red neuronal
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Paso 3 - Flattening: Aplanado de los datos
classifier.add(Flatten())

# Paso 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compilamos la CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Parte 2 - Ajustar la CNN a las imagenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit(training_set,
                         steps_per_epoch = training_set.n // training_set.batch_size,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = test_set.n // test_set.batch_size)


# Parte 3 - Haciendo nuevas predicciones
# Dentro de la carpeta dataset esta otra carpeta llamada single_prediction, en ella tiene 2 imagenes, 1 perro y 1 gato
# Necesitaremos numpy
import numpy as np
# Para cargar una sola imagen necestiaremos cargar image de Keras
from keras.preprocessing import image
# Debemos cargar la imagen, partiremos con la primera, indicandole la ruta como primer parametro el el tamaño objetivo de la imagen
test_image = image.load_img('dataset/single_prediction/mila.jpg', target_size = (64, 64))
# Transformaremos la imagen de tipo image a un array (64, 64, 3)
test_image = image.img_to_array(test_image)
# Debemos agregarle una dimencion adicional para que quede en total con 4 dimenciones
# Con axis 0 quedara como (1, 64, 64, 3) añadiendo la nueva dimencion en primer logar 
test_image = np.expand_dims(test_image, axis = 0)
# Ahora ya podemos hacer la prediccion con classifier.predict, pasandole la test_image y almacenandola en la variable result
result = classifier.predict(test_image)
# Si nos fijamos en la Variable Explorer en Spyder, el result nos dice que es 1, pero no sabemos si es perro o gato
# Para solucionarlo debemos mapearlo con el indice de la clase 
training_set.class_indices
# Al compilar esta linea vemos que nos devuelve por pantalla {'cats': 0, 'dogs': 1}
# Entonces ahora con un if lo podemos imprimir por pantalla 
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
# Al ejecutar este if... vemos que nos aparece en "Variable Explorer" de Spyder la variable prediction con value Dog