# Redes Neuronales Convolucionales

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

# Añadiremos una segunda capa de convolucion y maxPooling para mejorar la red neuronal
# Al Conv2D no le pasaremos el tamaño de entreda en el input_shape, ya que ahora no tienen ese tamaño
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Paso 3 - Flattening: Aplanado de los datos
classifier.add(Flatten())


# Paso 4 - Full connection
# Debemos añadirle una capa oculta con Dense, pasandole el tamaño de nodos, por ejemplo 128 y la activacion relu
classifier.add(Dense(units = 128, activation = 'relu'))
# Añadiremos otro mas, pero recuerda que la capa debe ser de salida con probabilidades (sigmoid) 
# Y solo una respuesta units = 1 (Clasificacion binario) perro o gato
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compilamos la CNN
# Primer parametro es el optimizador y eligeremos el metodo de adam
# el segundo es la funcion de perdida, utilizaremos la entropia binaria y no la categorica, ya que tenemos perros y gatos
# Y las metricas de precicion sera accuracy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Parte 2 - Ajustar la CNN a las imagenes para entrenar
# Importamos la libreria para el procesado de imagenes
from keras.preprocessing.image import ImageDataGenerator

# Con ImageDataGenerator le pasamos como parametro el rescale, para reescalar los pixeles de las imagenes
# con shear_range, zoom_range, horizontal_flip son parametros para mover las imagenes (Arriba, zoom, vertical, etc)
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Vamos a regenerar las imagenes pero en los test, y solo modificaremos el reescalado
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creamos el conjunto de entrenamiento, pasandole todas las imagenes para su entrenamiento 
# Debemos mantener el tamaño de 64x64
# El tamaño del bloque de carga seran de 32 lotes para poder catalogarse
# En class_mode sera binario ya que tenemos solo perros y gatos
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Creamos el conjunto de validacion (test), pasandole todas las imagenes para validar 
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Ahora es el turno de que el generador realice el ajuste del modelo con las imagenes del conjunto de entrenamiento
# steps_per_epoch, son los numeros de pasos por cada epoch, si tenemos 8.000 imagenes, sera este el valor a pasar
# con epochs es el numero de entrenamiento por imagenes
# validation_data sera el conjunto de imagenes de validacion
# Y por ultimo es el numero de paso de validacion, le pasaremos 2.000 validaciones
classifier.fit(training_set,
                         steps_per_epoch = training_set.n // training_set.batch_size,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = test_set.n // test_set.batch_size)
