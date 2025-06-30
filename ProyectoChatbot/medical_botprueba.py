# medical_bot_api.py
# Este archivo combina la lógica de entrenamiento, reentrenamiento y la API en un solo lugar.
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import os
import numpy as np
from flask import Flask, request, jsonify
import threading # Para ejecutar tareas en segundo plano

# --- CONFIGURACIÓN ---
MODEL_DIR = "modelo_medico"
DB_NAME = "MedicalBot.db"
EPOCHS_INICIAL = 30 # Ajusta según sea necesario
EPOCHS_REENTRENAMIENTO = 10 # Ajusta según sea necesario
BATCH_SIZE = 8 # Ajusta según sea necesario
REENTRENAMIENTO_THRESHOLD = 50 # Número de consultas para reentrenar

# --- Inicialización de la Aplicación Flask ---
app = Flask(__name__)

# --- VARIABLES GLOBALES (accesibles desde la API y otras funciones) ---
# Usamos variables globales para que el modelo, tokenizer y encoder
# se carguen una sola vez y estén disponibles para las peticiones.
modelo = None
tokenizer = None
label_encoder = None
df_original = None

# --- FUNCIONES DE LÓGICA DEL BOT (Iguales a las de tu script original) ---

def reentrenar_modelo():
    """
    Función para cargar datos de la BBDD, combinarlos con los originales
    y reentrenar el modelo. Esta función ahora actualiza las variables globales.
    """
    global modelo, tokenizer, label_encoder
    
    print("Iniciando proceso de reentrenamiento...")
    try:
        conn_re = sqlite3.connect(DB_NAME)
        nueva_data_raw = pd.read_sql("SELECT Sintoma, Enfermedad FROM Aprendizaje", conn_re)
        conn_re.close()

        if nueva_data_raw.empty:
            print("No hay nuevos datos para reentrenar.")
            return

        print(f"Reentrenando con {len(nueva_data_raw)} nuevos registros.")
        
        nueva_data = nueva_data_raw.rename(columns={'Sintoma': 'symptoms', 'Enfermedad': 'name'})
        data_completa = pd.concat([df_original[['symptoms', 'name']], nueva_data], ignore_index=True)

        X_nuevo = data_completa['symptoms'].tolist()
        y_text_nuevo = data_completa['name'].tolist()
        
        # Re-ajustar el LabelEncoder y guardarlo en la variable global
        label_encoder_nuevo = LabelEncoder()
        y_nuevo = label_encoder_nuevo.fit_transform(y_text_nuevo)
        num_labels_nuevo = len(label_encoder_nuevo.classes_)
        label_encoder = label_encoder_nuevo # ¡Actualización crucial!
        
        # Cargar y compilar el nuevo modelo
        tokenizer_nuevo = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        tokens_nuevo = tokenizer_nuevo(X_nuevo, truncation=True, padding=True, return_tensors='tf')
        
        model_nuevo = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=num_labels_nuevo
        )
        model_nuevo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
        
        model_nuevo.fit([tokens_nuevo['input_ids'], tokens_nuevo['attention_mask']], y_nuevo, epochs=EPOCHS_REENTRENAMIENTO, batch_size=BATCH_SIZE)

        # Guardar el modelo y tokenizer nuevos y actualizarlos en las variables globales
        model_nuevo.save_pretrained(MODEL_DIR)
        tokenizer_nuevo.save_pretrained(MODEL_DIR)
        modelo = model_nuevo
        tokenizer = tokenizer_nuevo
        
        print("✔ Modelo reentrenado y guardado exitosamente.")
        # Opcional: Limpiar la tabla de aprendizaje después de reentrenar
        # conn_clean = sqlite3.connect(DB_NAME)
        # conn_clean.cursor().execute("DELETE FROM Aprendizaje")
        # conn_clean.commit()
        # conn_clean.close()

    except Exception as e:
        print(f"Ocurrió un error durante el reentrenamiento: {e}")

def inicializar_y_entrenar():
    """
    Esta función contiene toda la lógica de preparación de datos,
    entrenamiento inicial y carga del modelo. Se llama una vez al iniciar la API.
    """
    global modelo, tokenizer, label_encoder, df_original
    
    print("Cargando y preparando los datos iniciales...")
    df_raw = pd.read_csv("enfermedades.csv")
    df = df_raw.rename(columns={
        'Enfermedad': 'name', 'Abstract': 'abstract', 'Tratamiento': 'treatment', 'Diagnostico': 'symptoms'
    })
    df = df[['symptoms', 'name', 'abstract', 'treatment']].dropna()
    df = df[df['symptoms'].str.len() > 5]
    if df.empty:
        raise ValueError("El DataFrame está vacío. Revisa tu archivo enfermedades.csv.")
    df_original = df.copy() # Guardamos una copia limpia para reentrenamientos

    print(f"Datos cargados. Se encontraron {len(df_original)} registros válidos.")

    # Crear la base de datos si no existe
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS Aprendizaje (Id INTEGER PRIMARY KEY, Sintoma TEXT, Enfermedad TEXT)")
    conn.commit()
    conn.close()
    
    # 3. Entrenamiento inicial del modelo (solo si no existe)
    if not os.path.exists(MODEL_DIR):
        print("No se encontró un modelo pre-entrenado. Iniciando entrenamiento inicial...")
        X = df_original['symptoms'].tolist()
        y_text = df_original['name'].tolist()
        
        # Ajustamos el LabelEncoder y lo guardamos en la variable global
        label_encoder_global = LabelEncoder()
        y = label_encoder_global.fit_transform(y_text)
        label_encoder = label_encoder_global
        
        num_labels = len(label_encoder.classes_)
        tokenizer_global = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        
        # ... (código de entrenamiento idéntico al original) ...
        X_tokenized = tokenizer_global(X, truncation=True, padding=True, max_length=512, return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices(((X_tokenized['input_ids'], X_tokenized['attention_mask']), y))
        train_size = int(0.8 * len(X))
        dataset = dataset.shuffle(buffer_size=len(X))
        train_dataset = dataset.take(train_size).batch(BATCH_SIZE)
        test_dataset = dataset.skip(train_size).batch(BATCH_SIZE)
        
        modelo_global = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        modelo_global.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        modelo_global.fit(train_dataset, epochs=EPOCHS_INICIAL, validation_data=test_dataset)
        
        modelo_global.save_pretrained(MODEL_DIR)
        tokenizer_global.save_pretrained(MODEL_DIR)
        modelo = modelo_global
        tokenizer = tokenizer_global
        print("✔ Modelo guardado.")
    else:
        print("Cargando modelo y tokenizer existentes...")
        modelo = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        
        # ¡CRUCIAL PARA LA PRECISIÓN!
        # Re-creamos el LabelEncoder a partir de TODOS los datos (originales + BBDD)
        # para que coincida con el estado actual del modelo guardado.
        conn_re = sqlite3.connect(DB_NAME)
        try:
            nuevos_datos = pd.read_sql("SELECT Sintoma, Enfermedad FROM Aprendizaje", conn_re)
            if not nuevos_datos.empty:
                nuevos_datos = nuevos_datos.rename(columns={'Sintoma': 'symptoms', 'Enfermedad': 'name'})
                datos_completos = pd.concat([df_original[['symptoms', 'name']], nuevos_datos], ignore_index=True)
                y_text_completos = datos_completos['name'].tolist()
                label_encoder_global = LabelEncoder()
                label_encoder_global.fit(y_text_completos)
                label_encoder = label_encoder_global
            else: # Si no hay datos nuevos, ajustamos solo con los originales
                y_text_original = df_original['name'].tolist()
                label_encoder_global = LabelEncoder()
                label_encoder_global.fit(y_text_original)
                label_encoder = label_encoder_global
        finally:
            conn_re.close()

        print("✔ Modelo y LabelEncoder sincronizados cargados.")


# --- ENDPOINTS DE LA API ---

@app.route('/predict', methods=['POST'])
def predict():
    if not all([modelo, tokenizer, label_encoder, df_original is not None]):
        return jsonify({"error": "El modelo no está listo. Revisa los logs del servidor."}), 500

    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({"error": "Faltan los síntomas en la petición."}), 400

    sintoma = data['symptoms']
    
    try:
        tokens = tokenizer(sintoma, return_tensors="tf", truncation=True, padding=True, max_length=512)
        logits = modelo(**tokens).logits
        predicted_class_idx = tf.argmax(logits, axis=1).numpy()[0]
        nombre_enfermedad = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        info = df_original[df_original['name'] == nombre_enfermedad]
        resumen = info.iloc[0]['abstract'] if not info.empty else "No se encontró resumen."
        tratamiento = info.iloc[0]['treatment'] if not info.empty else "No se encontró tratamiento."

        # Guardar en la base de datos para aprendizaje futuro
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Aprendizaje (Sintoma, Enfermedad) VALUES (?, ?)", (sintoma, nombre_enfermedad))
        conn.commit()
        
        # Comprobar si se debe reentrenar (en un hilo separado para no bloquear la respuesta)
        nuevos = cursor.execute("SELECT COUNT(*) FROM Aprendizaje").fetchone()[0]
        conn.close()
        
        if nuevos > 0 and nuevos % REENTRENAMIENTO_THRESHOLD == 0:
            print(f"\nSe ha alcanzado el umbral de {REENTRENAMIENTO_THRESHOLD} nuevos registros. Iniciando reentrenamiento en segundo plano...")
            # Usamos un hilo para que la respuesta al usuario sea inmediata
            thread = threading.Thread(target=reentrenar_modelo)
            thread.start()
            
        return jsonify({
            "predicted_disease": nombre_enfermedad,
            "summary": resumen,
            "treatment": tratamiento
        })
            
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error durante la predicción: {str(e)}"}), 500

# --- INICIO DE LA APLICACIÓN ---
if __name__ == '__main__':
    # 1. Ejecutar la inicialización y entrenamiento (si es necesario)
    inicializar_y_entrenar()
    
    # 2. Iniciar el servidor Flask
    print("\n* Iniciando servidor Flask. Listo para recibir peticiones en http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)