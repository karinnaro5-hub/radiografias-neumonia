import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image


# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

# Ruta del modelo entrenado.
ruta_modelo = "cnn_radiografias.keras"

# Ruta de la radiografía que queremos probar.
ruta_imagen = "radiografia_prueba.png"

# Tamaño utilizado durante el entrenamiento.
tamaño_imagen = (128, 128)


# ------------------------------------------------------------
# CARGAR EL MODELO
# ------------------------------------------------------------

modelo = keras.models.load_model(
    ruta_modelo
)

print("MODELO CARGADO CORRECTAMENTE")


# ------------------------------------------------------------
# CARGAR LA IMAGEN
# ------------------------------------------------------------

# Carga la imagen y la redimensiona a 128 × 128 píxeles.
imagen_prueba = image.load_img(
    ruta_imagen,
    target_size=tamaño_imagen
)


# ------------------------------------------------------------
# CONVERTIR LA IMAGEN EN ARRAY
# ------------------------------------------------------------

# Convierte la imagen en una matriz de números.
#
# La forma será:
#
# (128, 128, 3)
#
# 128 píxeles de alto
# 128 píxeles de ancho
# 3 canales RGB

imagen_array = image.img_to_array(
    imagen_prueba
)


# ------------------------------------------------------------
# NORMALIZAR LOS PÍXELES
# ------------------------------------------------------------

# Durante el entrenamiento utilizamos:
#
# rescale = 1./255
#
# Por eso, al probar el modelo, debemos aplicar
# exactamente la misma transformación.
#
# Los píxeles pasan del rango:
#
# [0,255]
#
# al rango:
#
# [0,1]

imagen_array = imagen_array / 255.0


# ------------------------------------------------------------
# AÑADIR LA DIMENSIÓN DEL BATCH
# ------------------------------------------------------------

# El modelo espera recibir grupos de imágenes.
#
# Antes:
#
# (128, 128, 3)
#
# Después:
#
# (1, 128, 128, 3)
#
# El número 1 indica que estamos enviando una sola imagen.

imagen_array = np.expand_dims(
    imagen_array,
    axis=0
)


# ------------------------------------------------------------
# HACER LA PREDICCIÓN
# ------------------------------------------------------------

prediccion = modelo.predict(
    imagen_array,
    verbose=0
)

print("\nPREDICCIÓN COMPLETA:")
print(prediccion)


# ------------------------------------------------------------
# EXTRAER EL RESULTADO
# ------------------------------------------------------------

# Como la última capa utiliza sigmoid,
# el modelo devuelve un único número entre 0 y 1.

probabilidad = float(
    prediccion[0][0]
)

print("\nVALOR DEVUELTO POR EL MODELO:")
print(probabilidad)


# ------------------------------------------------------------
# INTERPRETAR LA PREDICCIÓN
# ------------------------------------------------------------
#
# Debes comprobar el orden de las clases mostrado
# durante el entrenamiento:
#
# print(train_gen.class_indices)
#
# Por ejemplo:
#
# {'Normal': 0, 'Viral Pneumonia': 1}
#
# En ese caso:
#
# valores cercanos a 0 → Normal
# valores cercanos a 1 → NormalViral


if probabilidad >= 0.5:

    confianza = probabilidad * 100

    print("\nRESULTADO: VIRAL PNEUMONIA")
    print(
        "CONFIANZA:",
        round(confianza, 2),
        "%"
    )

else:

    confianza = (1 - probabilidad) * 100

    print("\nRESULTADO: NORMAL")
    print(
        "CONFIANZA:",
        round(confianza, 2),
        "%"
    )