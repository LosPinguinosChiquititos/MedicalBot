import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import os
import numpy as np

# 1. Cargar dataset original
df = pd.read_csv("enfermedades.csv")
df = df[['symptoms', 'name', 'abstract', 'treatment']].dropna()
df = df[df['symptoms'].str.len() > 5]

# 2. Preparar datos
X = df['symptoms'].tolist()
y = LabelEncoder().fit_transform(df['name'])
label_encoder = LabelEncoder()
label_encoder.fit(df['name'])

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
X_tokenized = tokenizer(X, truncation=True, padding=True, return_tensors='tf')
X_input_ids = X_tokenized['input_ids']
X_attention_mask = X_tokenized['attention_mask']

# 3. Entrenamiento inicial del modelo
X_input_ids_np = X_input_ids.numpy()
y_np = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_input_ids_np, y_np, test_size=0.2, random_state=42)

model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=len(set(y))
)
model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)

# 4. Guardar modelo
model.save_pretrained("modelo_medico")
tokenizer.save_pretrained("modelo_medico")

# 5. Conexión a base de datos
conn = sqlite3.connect("MedicalBot.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS Pacientes (
    IdPaciente INTEGER PRIMARY KEY AUTOINCREMENT,
    Nombre TEXT,
    Edad INTEGER,
    FechaConsulta DATETIME
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS Consultas (
    IdConsulta INTEGER PRIMARY KEY AUTOINCREMENT,
    IdPaciente INTEGER,
    Sintomas TEXT,
    Alergias TEXT,
    ActividadesPrevias TEXT,
    EnfermedadesPosibles TEXT,
    Recomendaciones TEXT,
    FechaConsulta DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (IdPaciente) REFERENCES Pacientes(IdPaciente)
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS Aprendizaje (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Sintoma TEXT,
    Enfermedad TEXT,
    FechaRegistro DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Función para reentrenar modelo
def reentrenar_modelo():
    nueva_data = pd.read_sql("SELECT Sintoma, Enfermedad FROM Aprendizaje", conn)
    if nueva_data.empty:
        print("No hay nuevos datos para reentrenar.")
        return

    original_data = df[['symptoms', 'name']].rename(columns={'symptoms': 'Sintoma', 'name': 'Enfermedad'})
    data_completa = pd.concat([original_data, nueva_data], ignore_index=True)

    X_nuevo = data_completa['Sintoma'].tolist()
    y_nuevo = LabelEncoder().fit_transform(data_completa['Enfermedad'])
    tokenizer_nuevo = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokens_nuevo = tokenizer_nuevo(X_nuevo, truncation=True, padding=True, return_tensors='tf')

    model_nuevo = TFDistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(set(y_nuevo))
    )
    model_nuevo.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
    model_nuevo.fit(tokens_nuevo['input_ids'], y_nuevo, epochs=50, batch_size=16)

    # Guardar nuevo modelo
    model_nuevo.save_pretrained("modelo_medico")
    tokenizer_nuevo.save_pretrained("modelo_medico")
    print("✔ Modelo reentrenado exitosamente.")

# 6. Interacción con el usuario
print("Bienvenido a MedicalBot. Usamos inteligencia artificial para ayudarte.")
nombre = input("Nombre: ")
edad = int(input("Edad: "))
fecha_consulta = datetime.now()
cursor.execute("INSERT INTO Pacientes (Nombre, Edad, FechaConsulta) VALUES (?, ?, ?)", (nombre, edad, fecha_consulta))
id_paciente = cursor.lastrowid

print("\nSelecciona el tipo de malestar:")
malestares = {"A": "Dolor de cabeza", "B": "Dolor estomacal", "C": "Fiebre", "D": "Náuseas", "E": "Tos", "F": "OTROS"}
for key, val in malestares.items():
    print(f"{key}. {val}")
opcion = input("Opción: ").upper()
sintoma = input("Describe tu síntoma: ") if opcion == "F" else input(f"¿Qué tipo de {malestares[opcion]}?: ")

alergias = input("¿Tienes alergias? (separadas por coma): ")
actividades = input("¿Qué hiciste antes del malestar?: ")

# 7. Cargar modelo entrenado
modelo = TFDistilBertForSequenceClassification.from_pretrained("modelo_medico")
tokenizer = DistilBertTokenizerFast.from_pretrained("modelo_medico")

# 8. Predicción
tokens = tokenizer(sintoma, return_tensors="tf", truncation=True, padding=True)
logits = modelo(**tokens).logits
predicted_class = tf.argmax(logits, axis=1).numpy()[0]
nombre_enfermedad = label_encoder.inverse_transform([predicted_class])[0]

info = df[df['name'] == nombre_enfermedad].iloc[0]
resumen = info['abstract']
tratamiento = info['treatment']

# 9. Verificar alergias
alergia_detectada = any(a.strip().lower() in tratamiento.lower() for a in alergias.split(','))
recomendacion = "⚠ Atención: posible contraindicación con tus alergias. Consulta a un médico." if alergia_detectada else tratamiento

# 10. Mostrar resultado
print("\n📋 Resultado:")
print("Enfermedad probable:", nombre_enfermedad)
print("Resumen:", resumen)
print("Recomendación:", recomendacion)

# 11. Guardar en base de datos
cursor.execute("""
INSERT INTO Consultas (IdPaciente, Sintomas, Alergias, ActividadesPrevias, EnfermedadesPosibles, Recomendaciones)
VALUES (?, ?, ?, ?, ?, ?)""", (
    id_paciente, sintoma, alergias, actividades, nombre_enfermedad, recomendacion
))
cursor.execute("INSERT INTO Aprendizaje (Sintoma, Enfermedad) VALUES (?, ?)", (sintoma, nombre_enfermedad))
conn.commit()

# 12. Reentrenar si hay 50 registros nuevos
nuevos = cursor.execute("SELECT COUNT(*) FROM Aprendizaje").fetchone()[0]
if nuevos >= 50:
    print("\n🔄 Reentrenando modelo con nuevos datos acumulados...")
    reentrenar_modelo()

conn.close()