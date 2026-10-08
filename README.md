# Radiografías Neumonía

Clasificador de radiografías de tórax con una red neuronal convolucional (CNN) entrenada desde cero en Keras/TensorFlow. Distingue entre dos clases, **Normal** y **Viral Pneumonia**, e incluye una interfaz web en Flask para subir una imagen y ver el resultado.

> **Aviso:** proyecto con fines educativos. El resultado lo genera un modelo automático y no reemplaza el diagnóstico de un profesional médico.

## Estructura

```
radiografias-neumonia/
├── cnn.py                      # Arquitectura y compilación de la CNN
├── 2_TrainDataAumegtation.py   # Entrenamiento con data augmentation
├── 3_test_modelo.py            # Prueba del modelo con una sola imagen
├── app.py                      # Servidor Flask (interfaz web + endpoint /predict)
├── cnn_radiografias.keras      # Modelo ya entrenado
├── radiografia_prueba.png      # Radiografía de ejemplo
├── requirements.txt
├── templates/index.html        # Página principal
└── static/                     # CSS y JavaScript de la interfaz
```

## Instalación

Requiere Python 3 con soporte para TensorFlow.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Uso

### Interfaz web

```bash
python app.py
```

Abre http://127.0.0.1:5000 en el navegador, arrastra o selecciona una radiografía (PNG o JPG, máx. 5 MB) y pulsa **Analizar radiografía**. Se muestra la clase detectada, el porcentaje de confianza y el valor bruto que devuelve el modelo.

El endpoint también puede usarse directamente:

```bash
curl -F "imagen=@radiografia_prueba.png" http://127.0.0.1:5000/predict
```

Respuesta:

```json
{ "resultado": "Normal", "confianza": 97.3, "probabilidad_bruta": 0.973 }
```

### Probar el modelo desde la terminal

```bash
python 3_test_modelo.py
```

Analiza `radiografia_prueba.png`. Para probar otra imagen, cambia `ruta_imagen` en el script.

### Reentrenar el modelo

El dataset no está incluido en el repositorio. Colócalo en una carpeta `dataset/` con una subcarpeta por clase:

```
dataset/
├── Normal/
└── Viral Pneumonia/
```

Después ejecuta:

```bash
python 2_TrainDataAumegtation.py
```

El script imprime el orden de clases que asignó Keras (`class_indices`), entrena 20 épocas y sobrescribe `cnn_radiografias.keras`. Si el orden de clases cambia, actualiza `ETIQUETAS` en `app.py` y la interpretación en `3_test_modelo.py`.

## Modelo

- **Entrada:** imagen RGB de 128 × 128 píxeles, normalizada a [0, 1].
- **Arquitectura:** 3 bloques Conv2D + MaxPooling (32, 64 y 128 filtros), Flatten, Dense de 128 con ReLU, Dropout 0.5 y una neurona de salida con sigmoide.
- **Entrenamiento:** optimizador Adam, pérdida `binary_crossentropy`, 80 % / 20 % de división entrenamiento/validación.
- **Data augmentation:** rotación de hasta 20°, desplazamientos del 10 %, zoom del 10 % y reflejo horizontal (solo en entrenamiento).
- **Interpretación de la salida:** un valor ≥ 0.5 se clasifica como **Normal**; un valor < 0.5, como **Viral Pneumonia**.

## Limitaciones

- Solo reconoce dos clases: no detecta neumonía bacteriana, COVID-19 ni otras patologías.
- Una imagen que no sea una radiografía de tórax recibirá igualmente una de las dos etiquetas.
- El porcentaje de confianza es la salida de la sigmoide, no una probabilidad clínica calibrada.
