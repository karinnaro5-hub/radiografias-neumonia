import io
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow import keras

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "cnn_radiografias.keras")
TAMANO_IMAGEN = (128, 128)
EXTENSIONES_PERMITIDAS = {"png", "jpg", "jpeg"}

# Orden de clases usado durante el entrenamiento.
# Invertido respecto al comentario original de 3_test_modelo.py:
# confirmado en uso real que probabilidad >= 0.5 corresponde a "Normal".
ETIQUETAS = {0: "Viral Pneumonia", 1: "Normal"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB por imagen

modelo = keras.models.load_model(RUTA_MODELO)


def extension_permitida(nombre_archivo):
    return (
        "." in nombre_archivo
        and nombre_archivo.rsplit(".", 1)[1].lower() in EXTENSIONES_PERMITIDAS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "imagen" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen."}), 400

    archivo = request.files["imagen"]

    if archivo.filename == "":
        return jsonify({"error": "No se seleccionó ningún archivo."}), 400

    if not extension_permitida(archivo.filename):
        return jsonify({"error": "Formato no soportado. Usa PNG o JPG."}), 400

    try:
        imagen = Image.open(io.BytesIO(archivo.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "No se pudo leer la imagen enviada."}), 400

    imagen = imagen.resize(TAMANO_IMAGEN)
    imagen_array = np.array(imagen, dtype="float32") / 255.0
    imagen_array = np.expand_dims(imagen_array, axis=0)

    prediccion = modelo.predict(imagen_array, verbose=0)
    probabilidad = float(prediccion[0][0])

    if probabilidad >= 0.5:
        etiqueta = ETIQUETAS[1]
        confianza = probabilidad * 100
    else:
        etiqueta = ETIQUETAS[0]
        confianza = (1 - probabilidad) * 100

    return jsonify(
        {
            "resultado": etiqueta,
            "confianza": round(confianza, 2),
            "probabilidad_bruta": round(probabilidad, 4),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
