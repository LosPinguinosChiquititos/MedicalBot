# api_medica.py
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import numpy as np
import os

# --- Configuración ---
MODEL_DIR = "modelo_medico"

# --- Inicialización de la Aplicación Flask ---
app = Flask(__name__)

# --- Cargar el Modelo y los Datos (ESTO SE HACE UNA SOLA VEZ AL INICIAR) ---
print("Cargando recursos del modelo. Esto puede tardar un momento...")

# Cargar el dataframe original para obtener información de la enfermedad
try:
    df_raw = pd.read_csv("enfermedades.csv")
    df = df_raw.rename(columns={
        'Enfermedad': 'name',
        'Abstract': 'abstract',
        'Tratamiento': 'treatment',
        'Diagnostico': 'symptoms'
    })
    df = df[['name', 'abstract', 'treatment']].dropna()

    # Cargar el modelo de IA y el tokenizador
    modelo = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)

    # Cargar el LabelEncoder (necesitamos recrearlo para poder usar inverse_transform)
    # Esto asume que las enfermedades en el df original son todas las posibles etiquetas
    y_text = df_raw['Enfermedad'].dropna().tolist() # Usamos el df_raw para tener la lista completa
    label_encoder = LabelEncoder()
    label_encoder.fit(y_text)
    
    print("✔ Modelo y recursos cargados exitosamente.")
    model_loaded = True
except Exception as e:
    print(f"Error al cargar el modelo o los datos: {e}")
    model_loaded = False
    df = None
    modelo = None
    tokenizer = None
    label_encoder = None

# --- Definir el "Endpoint" de la API ---
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({"error": "El modelo no está disponible. Revisa los logs del servidor."}), 500

    # Obtener los datos JSON enviados en la petición
    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({"error": "Faltan los síntomas en la petición."}), 400

    sintoma = data['symptoms']
    
    # Realizar la predicción
    try:
        tokens = tokenizer(sintoma, return_tensors="tf", truncation=True, padding=True, max_length=512)
        logits = modelo(**tokens).logits
        predicted_class_idx = tf.argmax(logits, axis=1).numpy()[0]
        nombre_enfermedad = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Buscar información adicional de la enfermedad
        info = df[df['name'] == nombre_enfermedad]
        if not info.empty:
            resumen = info.iloc[0]['abstract']
            tratamiento = info.iloc[0]['treatment']
        else:
            resumen = "No se encontró información adicional para esta enfermedad."
            tratamiento = "Consulte a un médico para obtener recomendaciones de tratamiento."

        # Crear la respuesta en formato JSON
        response = {
            "predicted_disease": nombre_enfermedad,
            "summary": resumen,
            "treatment": tratamiento
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error durante la predicción: {str(e)}"}), 500

# --- Ejecutar la aplicación ---
if __name__ == '__main__':
    # Escuchará en todas las interfaces de red en el puerto 5000
    app.run(host='0.0.0.0', port=5000, debug=True)