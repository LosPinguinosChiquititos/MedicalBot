# Este archivo combina la lógica de entrenamiento, carga del modelo, la API Flask
# y la conexión a una base de datos SQL Server.

import pandas as pd
import pyodbc  # Driver para conectar con SQL Server
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import os
import numpy as np
from flask import Flask, request, jsonify
import threading

# ==============================================================================
# --- CONFIGURACIÓN ---
# ==============================================================================

# --- Configuración del Modelo ---
MODEL_DIR = "modelo_medico"
EPOCHS_INICIAL = 100  #<---- Esto solo se utilizó en la creación del modelo asi que se omite después de haberlo creado en /modelo-medico
EPOCHS_REENTRENAMIENTO = 50
BATCH_SIZE = 8
REENTRENAMIENTO_THRESHOLD = 10  # Reentrenar cada 50 nuevas consultas

# --- Configuración de la Base de Datos SQL Server ---
# ¡¡¡IMPORTANTE!!! DEBES AJUSTAR ESTOS VALORES A TU CONFIGURACIÓN
# ------------------------------------------------------------------------------
DB_SERVER = 'localhost'  # Ej: 'LAPTOP-MIA\SQLEXPRESS', 'localhost', o el nombre del servidor remoto
DB_DATABASE = 'MedicalBot'          # El nombre de tu base de datos

# --- Elige la CADENA DE CONEXIÓN que se ajuste a tu caso ---

# Opción A: Autenticación de Windows (la más común en desarrollo local)
# El servidor SQL confía en tu usuario de Windows. No necesita usuario/contraseña.
CONNECTION_STRING = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_DATABASE};Trusted_Connection=yes;'

# Opción B: Autenticación de SQL Server
# Usas un usuario y contraseña creados específicamente en SQL Server.
# CONNECTION_STRING = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_DATABASE};UID={DB_USER};PWD={DB_PASSWORD};'

# ==============================================================================
# --- VARIABLES GLOBALES Y APLICACIÓN FLASK ---
# ==============================================================================
app = Flask(__name__)
modelo, tokenizer, label_encoder, df_original = None, None, None, None

# ==============================================================================
# --- LÓGICA DE MODELO Y ENTRENAMIENTO ---
# ==============================================================================

def reentrenar_modelo():
    """
    Carga datos nuevos desde la tabla 'Consultas' de SQL Server, los combina
    con los originales y reentrena el modelo, actualizando las variables globales.
    """
    global modelo, tokenizer, label_encoder
    
    print("Iniciando proceso de reentrenamiento en segundo plano...")
    try:
        conn = pyodbc.connect(CONNECTION_STRING)
        query = "SELECT Sintomas, EnfermedadesPosibles FROM Consultas WHERE Sintomas IS NOT NULL AND EnfermedadesPosibles IS NOT NULL"
        nueva_data_raw = pd.read_sql(query, conn)
        conn.close()

        if nueva_data_raw.empty:
            print("No hay nuevos datos en 'Consultas' para reentrenar.")
            return

        print(f"Reentrenando con {len(nueva_data_raw)} registros de consultas.")
        
        nueva_data = nueva_data_raw.rename(columns={'Sintomas': 'symptoms', 'EnfermedadesPosibles': 'name'})
        data_completa = pd.concat([df_original[['symptoms', 'name']], nueva_data], ignore_index=True)

        X_nuevo = data_completa['symptoms'].tolist()
        y_text_nuevo = data_completa['name'].tolist()
        
        # Re-ajustar el LabelEncoder y guardarlo en la variable global
        label_encoder_nuevo = LabelEncoder()
        y_nuevo = label_encoder_nuevo.fit_transform(y_text_nuevo)
        num_labels_nuevo = len(label_encoder_nuevo.classes_)
        label_encoder = label_encoder_nuevo  # Actualización crucial
        
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

        # Guardar y actualizar las variables globales
        model_nuevo.save_pretrained(MODEL_DIR)
        tokenizer_nuevo.save_pretrained(MODEL_DIR)
        modelo = model_nuevo
        tokenizer = tokenizer_nuevo
        
        print("✔ Modelo reentrenado y guardado exitosamente.")

    except Exception as e:
        print(f"Ocurrió un error durante el reentrenamiento: {e}")

def inicializar_aplicacion():
    """
    Esta función ahora puede:
    1. Entrenar el modelo desde cero si 'modelo_medico' no existe.
    2. Cargar un modelo existente y sincronizar el LabelEncoder.
    """
    global modelo, tokenizer, label_encoder, df_original
    
    print("Cargando y preparando los datos iniciales del archivo CSV...")
    try:
        # Aquí puedes añadir la lógica para usar la columna 'Sintomas_Paciente' si la creaste
        df_raw = pd.read_csv("enfermedades.csv")
        df = df_raw.rename(columns={'Enfermedad': 'name', 'Abstract': 'abstract', 'Tratamiento': 'treatment', 'Diagnostico': 'symptoms'})
        df_original = df[['symptoms', 'name', 'abstract', 'treatment']].dropna()
    except FileNotFoundError:
        raise RuntimeError("El archivo 'enfermedades.csv' no se encontró. Asegúrate de que esté en la misma carpeta que el script.")

    # --- INICIO DEL BLOQUE DE CÓDIGO QUE HAS PEGADO, AHORA INTEGRADO ---
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
        
        # Código de entrenamiento idéntico al original
        X_tokenized = tokenizer_global(X, truncation=True, padding=True, max_length=512, return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices(((X_tokenized['input_ids'], X_tokenized['attention_mask']), y))
        train_size = int(0.8 * len(X))
        dataset = dataset.shuffle(buffer_size=len(X))
        train_dataset = dataset.take(train_size).batch(BATCH_SIZE)
        test_dataset = dataset.skip(train_size).batch(BATCH_SIZE)
        
        modelo_global = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        modelo_global.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        
        # Usamos la constante EPOCHS_INICIAL
        modelo_global.fit(train_dataset, epochs=EPOCHS_INICIAL, validation_data=test_dataset)
        
        modelo_global.save_pretrained(MODEL_DIR)
        tokenizer_global.save_pretrained(MODEL_DIR)
        
        # Asignamos los objetos recién entrenados a las variables globales
        modelo = modelo_global
        tokenizer = tokenizer_global
        print("✔ Modelo inicial entrenado y guardado.")
    # --- FIN DEL BLOQUE DE CÓDIGO INTEGRADO ---
        
    else:
        # Si el modelo ya existe, se ejecuta la lógica de carga que ya tenías
        print("Cargando modelo y tokenizer existentes...")
        modelo = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        
        print("Sincronizando LabelEncoder con los datos actuales (CSV + Base de Datos)...")
        try:
            conn = pyodbc.connect(CONNECTION_STRING)
            query = "SELECT Sintomas, EnfermedadesPosibles FROM Consultas WHERE Sintomas IS NOT NULL AND EnfermedadesPosibles IS NOT NULL"
            nuevos_datos = pd.read_sql(query, conn)
            conn.close()
            
            nuevos_datos = nuevos_datos.rename(columns={'Sintomas': 'symptoms', 'EnfermedadesPosibles': 'name'})
            datos_completos = pd.concat([df_original[['symptoms', 'name']], nuevos_datos], ignore_index=True)
            y_text_completos = datos_completos['name'].unique().tolist()
            
            label_encoder_global = LabelEncoder()
            label_encoder_global.fit(y_text_completos)
            label_encoder = label_encoder_global
            
            print(f"✔ LabelEncoder sincronizado con {len(label_encoder.classes_)} enfermedades.")
            
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo conectar a la base de datos para sincronizar el LabelEncoder. Se usará solo el CSV. Error: {e}")
            y_text_original = df_original['name'].unique().tolist()
            label_encoder_global = LabelEncoder()
            label_encoder_global.fit(y_text_original)
            label_encoder = label_encoder_global
            print(f"✔ LabelEncoder cargado con {len(label_encoder.classes_)} enfermedades del archivo CSV.")

# ==============================================================================
# --- ENDPOINTS DE LA API (El "Servidor Web") ---
# ==============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Recibe los datos del paciente y los síntomas, realiza una predicción,
    comprueba las alergias, guarda la consulta en SQL Server y devuelve el resultado.
    """
    try:
        data = request.get_json()
        sintoma = data.get('symptoms')
        nombre_paciente = data.get('patientName')
        edad_paciente = data.get('patientAge')
        actividades_previas = data.get('previousActivities')
        alergias = data.get('allergies', '')  # Recibimos el campo de alergias

        if not all([sintoma, nombre_paciente, edad_paciente is not None]):
            return jsonify({"error": "Faltan datos obligatorios (nombre, edad, síntomas)."}), 400

        # --- 1. Realizar la Predicción ---
        tokens = tokenizer(sintoma, return_tensors="tf", truncation=True, padding=True, max_length=512)
        logits = modelo(**tokens).logits
        predicted_class_idx = tf.argmax(logits, axis=1).numpy()[0]
        nombre_enfermedad = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        info = df_original[df_original['name'] == nombre_enfermedad].iloc[0]
        resumen = info['abstract']
        tratamiento = info['treatment']
        
        # --- CAMBIO: LÓGICA DE VERIFICACIÓN DE ALERGIAS ---
        recomendacion_final = tratamiento # Empezamos con el tratamiento original
        if alergias.strip():
            # Convertimos todo a minúsculas para una comparación sin distinción de mayúsculas/minúsculas
            tratamiento_lower = tratamiento.lower()
            # Dividimos las alergias por si el usuario introduce varias separadas por coma
            lista_alergias = [a.strip().lower() for a in alergias.split(',')]
            
            alergia_detectada = False
            for alergia in lista_alergias:
                if alergia and alergia in tratamiento_lower: # Comprobamos que no sea una cadena vacía
                    alergia_detectada = True
                    break # Salimos del bucle si encontramos una coincidencia
            
            if alergia_detectada:
                # Si se detecta una alergia, añadimos una advertencia clara al tratamiento
                advertencia = (
                    "\n\n<strong style='color: red;'>⚠ ¡ADVERTENCIA DE ALERGIA!</strong> "
                    "El tratamiento sugerido podría contener sustancias a las que has indicado ser alérgico. "
                    "<strong>NO TOMES NINGÚN MEDICAMENTO sin consultar primero a un médico.</strong>"
                )
                recomendacion_final += advertencia
        
        # --- 2. Guardar en la Base de Datos SQL Server ---
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        
        sql_paciente = "INSERT INTO Pacientes (Nombre, Edad, FechaConsulta) OUTPUT INSERTED.IdPaciente VALUES (?, ?, GETDATE());"
        id_paciente = cursor.execute(sql_paciente, nombre_paciente, edad_paciente).fetchval()
        
        # Usamos la 'recomendacion_final' que puede incluir la advertencia
        sql_consulta = """
            INSERT INTO Consultas (IdPaciente, Sintomas, Alergias, ActividadesPrevias, EnfermedadesPosibles, Recomendaciones)
            VALUES (?, ?, ?, ?, ?, ?);
        """
        cursor.execute(sql_consulta, id_paciente, sintoma, alergias, actividades_previas, nombre_enfermedad, recomendacion_final)
        
        conn.commit()
        
        count_query = "SELECT COUNT(*) FROM Consultas"
        num_consultas = cursor.execute(count_query).fetchval()
        
        cursor.close()
        conn.close()

        print(f"Consulta guardada para paciente ID: {id_paciente}. Total de consultas: {num_consultas}.")

        if num_consultas > 0 and num_consultas % REENTRENAMIENTO_THRESHOLD == 0:
            thread = threading.Thread(target=reentrenar_modelo)
            thread.start()
            
        # --- 3. Devolver la Respuesta ---
        return jsonify({
            "predicted_disease": nombre_enfermedad,
            "summary": resumen,
            "treatment": recomendacion_final # Devolvemos la recomendación que ya incluye la advertencia si es necesario
        })
            
    except Exception as e:
        print(f"!!! ERROR EN /predict: {e}")
        return jsonify({"error": f"Ocurrió un error en el servidor Python: {str(e)}"}), 500

# ==============================================================================
# --- INICIO DE LA APLICACIÓN ---
# ==============================================================================
if __name__ == '__main__':
    # 1. Cargar el modelo y sincronizar el LabelEncoder
    inicializar_aplicacion()
    
    # 2. Iniciar el servidor Flask
    print("\n=========================================================")
    print(f"* Iniciando servidor Flask en modo {'DEBUG' if app.debug else 'PRODUCCIÓN'}")
    print(f"* Listo para recibir peticiones en http://127.0.0.1:5000")
    print("=========================================================")
    app.run(host='0.0.0.0', port=5000, debug=False) # Se recomienda debug=False para producción