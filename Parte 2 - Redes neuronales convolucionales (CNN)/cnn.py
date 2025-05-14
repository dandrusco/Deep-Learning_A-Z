# Redes Neuronales Convolucionales

# -----------------------------------------------------------------------------
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras
# -----------------------------------------------------------------------------

# Parte 1 - Construir el modelo de CNN

# Para que Keras aprenda, debemos tener preparado una carpeta llamada test_set y otra training_set
# Dentro de ambaas carpetas debemos tener una carpeta para gatos y otra para perros
# Dentro de cada sub-carpeta debe terner las imagenes segun coreespondan 

# Para validar (test) contamos con 1.000 imagenes de perros y 1.000 de gatos

# Para entrenar (training) contamos con 4.000 imaenes de gatos y 4.000 de perros 


# Importar librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializamos la CNN (Redes Neuronales Convolucionales)
classifier = Sequential()

# Paso 1 - Convolucion
# Al propio classifier le añadiremos una Convolucion en 2D
# Añadiremos el numero de filtro (detectores de caracteristicas) 32 (Potencia de 2), columnas y finas (3X3)
# input_shape es el tamaño de las imagenes 64 x 64 y el tercer parametro es el canal de color (rojo, verde, azul) 3
# activation corresponde a la activacion de las neuronas, utilizaremos relu
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Paso 2 - Max Pooling
# Añadiremos una nueva capa pero de Max Pooling, añadiendo una matriz, en este caso 2X2
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#--------------------------------------------------------------------
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#--------------------------------------------------------------------

# Paso 3 - Flattening: Aplanado de los datos
classifier.add(Flatten())

# Paso 4 - Full connection
# Debemos añadirle una capa oculta con Dense, pasandole el tamaño de nodos, por ejemplo 128 y la activacion relu
classifier.add(Dense(units = 128, activation = 'relu'))
# Añadiremos otro mas, pero recuerda que la capa debe ser de salida con probabilidades (sigmoid) 
# Y solo una respuesta units = 1 (Clasificacion binario) perro o gato
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

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

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)