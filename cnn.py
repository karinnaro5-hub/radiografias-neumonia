from tensorflow.keras import layers, models

# ------------------------------------------------------------
# CREAR UNA RED NEURONAL CONVOLUCIONAL (CNN)
# ------------------------------------------------------------
#
# Esta CNN aprenderá desde cero.
# A diferencia del Transfer Learning, aquí no utilizamos
# un modelo preentrenado como MobileNetV2.
#
# Será la propia red la que aprenderá las características
# de las imágenes durante el entrenamiento.
#

modelo = models.Sequential([

    # --------------------------------------------------------
    # CAPA DE ENTRADA
    # --------------------------------------------------------
    #
    # La red recibirá imágenes de:
    #
    # 128 píxeles de alto
    # 128 píxeles de ancho
    # 3 canales de color (RGB)

    layers.Input(shape=(128, 128, 3)),


    # ========================================================
    # BLOQUE CONVOLUCIONAL 1
    # ========================================================

    # Conv2D aplica filtros sobre la imagen.
    #
    # En este bloque se aprenderán características simples,
    # como:
    #
    # - bordes
    # - líneas
    # - colores
    # - texturas

    layers.Conv2D(

        filters=32,

        kernel_size=(3,3),

        activation="relu"

    ),

    # Reduce el tamaño de la imagen conservando
    # únicamente la información más importante.
    #
    # Esto acelera el entrenamiento
    # y reduce el número de parámetros.

    layers.MaxPooling2D((2,2)),


    # ========================================================
    # BLOQUE CONVOLUCIONAL 2
    # ========================================================

    # Ahora la red combina las características aprendidas
    # anteriormente para detectar patrones más complejos.

    layers.Conv2D(

        filters=64,

        kernel_size=(3,3),

        activation="relu"

    ),

    layers.MaxPooling2D((2,2)),


    # ========================================================
    # BLOQUE CONVOLUCIONAL 3
    # ========================================================

    # En este nivel la red puede aprender
    # características más abstractas.
    #
    # Por ejemplo:
    #
    # - ojos
    # - hocico
    # - orejas
    # - patas
    # - forma del animal

    layers.Conv2D(

        filters=128,

        kernel_size=(3,3),

        activation="relu"

    ),

    layers.MaxPooling2D((2,2)),


    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------
    #
    # Convierte todos los mapas de características
    # en una única lista de números.
    #
    # Esta lista será utilizada por las capas Dense.

    layers.Flatten(),


    # --------------------------------------------------------
    # CAPA DENSA
    # --------------------------------------------------------
    #
    # Combina toda la información extraída
    # por las capas convolucionales.

    layers.Dense(

        units=128,

        activation="relu"

    ),


    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------
    #
    # Durante el entrenamiento apaga aleatoriamente
    # el 50 % de las neuronas.
    #
    # Esto ayuda a evitar el sobreajuste
    # (overfitting).

    layers.Dropout(0.5),


    # --------------------------------------------------------
    # CAPA DE SALIDA
    # --------------------------------------------------------
    #
    # Tenemos únicamente dos clases:
    #
    # perro
    # gato
    #
    # Sigmoid devuelve un valor entre 0 y 1.

    layers.Dense(

        units=1,

        activation="sigmoid"

    )

])


# ------------------------------------------------------------
# CONFIGURAR EL MODELO
# ------------------------------------------------------------

modelo.compile(

    # Algoritmo encargado de actualizar
    # los pesos durante el entrenamiento.
    optimizer="adam",

    # Función de pérdida utilizada
    # en clasificación binaria.
    loss="binary_crossentropy",

    # Accuracy indica el porcentaje
    # de imágenes clasificadas correctamente.
    metrics=["accuracy"]

)


# Mostrar la estructura completa del modelo.
modelo.summary()