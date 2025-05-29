# Redes Neuronales Convolucionales

# Parte 1 - Construir el modelo de CNN

# Importar librerias y paquetes, añadiremos Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Inicializamos la CNN (Redes Neuronales Convolucionales)
classifier = Sequential()

# 1ra capa
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2da capa
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 3ra capa
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Paso 3 - Flattening: Aplanado de los datos
classifier.add(Flatten())

# Paso 4 - Full connection 
# Capa densa
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))  # Muy útil para evitar overfitting (aprendizaje “excesivo”)

# Capa de salida
classifier.add(Dense(units = 1, activation = 'sigmoid')) # Binario

# Compilamos la CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Parte 2 - Ajustar la CNN a las imagenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
                                )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

#Los callbacks ayudan a evitar sobreentrenamiento:
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Definimos los callbacks para detener temprano el entrenamiento y guardar el mejor modelo
# Si val_loss no mejora durante 8 épocas consecutivas, se detiene el entrenamiento automáticamente.
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint('mejor_modelo.h5', monitor='val_loss', save_best_only=True)


classifier.fit(training_set,
                         steps_per_epoch = training_set.n // training_set.batch_size,
                         epochs = 90,
                         validation_data = test_set,
                         validation_steps = test_set.n // test_set.batch_size,
                         callbacks=[early_stop, checkpoint])


# Parte 3 - Haciendo nuevas predicciones
import numpy as np
from keras.preprocessing import image

# Debemos cargar la imagen, partiremos con la primera, indicandole la ruta como primer parametro el el tamaño objetivo de la imagen
test_image = image.load_img('dataset/single_prediction/bruno.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'