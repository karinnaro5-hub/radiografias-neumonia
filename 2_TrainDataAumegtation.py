# IMPORTAR EL MODELO CREADO Y COMPILADO EN cnn.py
from cnn import modelo

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------

# Ruta donde están las carpetas de las clases.
#
# Por ejemplo:
#
# dataset/
# ├── Normal/
# └── Viral_Pneumonia/

ruta_dataset = "M2/Clase8/dataset/"

# Tamaño que espera la CNN.
tamaño_imagen = (128, 128)

# Número de imágenes que se procesan en cada lote.
tamaño_batch = 32

# Semilla para garantizar que entrenamiento y validación
# utilicen siempre la misma división de imágenes.
semilla = 42


# ------------------------------------------------------------
# GENERADOR DE ENTRENAMIENTO
# ------------------------------------------------------------
#
# En las imágenes de entrenamiento sí aplicamos
# Data Augmentation.
#
# Las transformaciones se generan durante el entrenamiento.
# Las imágenes originales no se modifican.

generador_entrenamiento = ImageDataGenerator(

    # Convierte los píxeles del rango [0,255]
    # al rango [0,1].
    rescale=1./255,

    # Rota aleatoriamente las imágenes
    # hasta un máximo de 20 grados.
    rotation_range=20,

    # Desplaza horizontalmente las imágenes
    # hasta un máximo del 10 %.
    width_shift_range=0.1,

    # Desplaza verticalmente las imágenes
    # hasta un máximo del 10 %.
    height_shift_range=0.1,

    # Aplica un zoom aleatorio de hasta el 10 %.
    zoom_range=0.1,

    # Refleja horizontalmente algunas imágenes.
    horizontal_flip=True,

    # Reserva el 20 % del dataset para validación.
    validation_split=0.20
)


# ------------------------------------------------------------
# GENERADOR DE VALIDACIÓN
# ------------------------------------------------------------
#
# En validación no aplicamos rotaciones, desplazamientos
# ni reflejos.
#
# Solo reescalamos los píxeles para que tengan
# el mismo rango que las imágenes de entrenamiento.

generador_validacion = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.20
)


# ------------------------------------------------------------
# CARGAR IMÁGENES DE ENTRENAMIENTO
# ------------------------------------------------------------

train_gen = generador_entrenamiento.flow_from_directory(

    # Carpeta principal del dataset.
    ruta_dataset,

    # Todas las imágenes se redimensionan a 128 × 128.
    target_size=tamaño_imagen,

    # Se procesan 32 imágenes en cada lote.
    batch_size=tamaño_batch,

    # Se utiliza clasificación binaria porque tenemos
    # exactamente dos clases.
    class_mode="binary",

    # Utiliza el 80 % reservado para entrenamiento.
    subset="training",

    # Debe coincidir con la semilla de validación
    # para que ambos generadores hagan la misma división.
    seed=semilla,

    # Mezcla las imágenes durante el entrenamiento.
    shuffle=True
)


# ------------------------------------------------------------
# CARGAR IMÁGENES DE VALIDACIÓN
# ------------------------------------------------------------

val_gen = generador_validacion.flow_from_directory(

    ruta_dataset,

    target_size=tamaño_imagen,

    batch_size=tamaño_batch,

    class_mode="binary",

    # Utiliza el 20 % reservado para validación.
    subset="validation",

    seed=semilla,

    # No es necesario mezclar las imágenes de validación.
    shuffle=False
)


# ------------------------------------------------------------
# COMPROBAR LAS CLASES
# ------------------------------------------------------------
#
# Muestra qué número ha asignado Keras a cada carpeta.
#
# Por ejemplo:
#
# {'Normal': 0, 'Viral Pneumonia': 1}

print("\nCLASES DETECTADAS:")
print(train_gen.class_indices)

print("\nIMÁGENES DE ENTRENAMIENTO:")
print(train_gen.samples)

print("\nIMÁGENES DE VALIDACIÓN:")
print(val_gen.samples)


# ------------------------------------------------------------
# ENTRENAR EL MODELO
# ------------------------------------------------------------

historial = modelo.fit(

    # Imágenes utilizadas para aprender.
    train_gen,

    # Imágenes utilizadas para comprobar el rendimiento
    # al finalizar cada época.
    validation_data=val_gen,

    # Número de veces que el modelo recorrerá
    # todas las imágenes de entrenamiento.
    epochs=20
)


# ------------------------------------------------------------
# GUARDAR EL MODELO
# ------------------------------------------------------------

modelo.save(
    "M2/Clase8/cnn_radiografias.keras"
)

print("\nMODELO GUARDADO CORRECTAMENTE")