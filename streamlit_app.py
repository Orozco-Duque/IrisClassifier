import streamlit as st
import pandas as pd
import joblib # Para cargar el modelo

# --- 1. Cargar el Modelo y Nombres de Características (usando st.cache_resource) ---
@st.cache_resource
def load_model():
    """Carga el modelo entrenado desde el archivo .pkl."""
    model = joblib.load('iris_model.pkl')
    return model

@st.cache_resource
def load_feature_names():
    """Carga los nombres de las características desde el archivo .pkl."""
    feature_names = joblib.load('iris_feature_names.pkl')
    return feature_names

model = load_model()
feature_names = load_feature_names()

# Mapeo de las predicciones numéricas a nombres de especies
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# --- 2. Título y Descripción de la Aplicación ---
st.title("Clasificador de Especies de Iris")
st.write("""
Esta aplicación predice la especie de una flor de Iris (Setosa, Versicolor o Virginica)
basándose en las medidas de sus sépalos y pétalos.
""")

# --- 3. Campos de Entrada para las Características ---
st.header("Introduce las Características de la Flor:")

# Creamos sliders para cada característica
# Usamos los nombres de las características cargados
sepal_length = st.slider(feature_names[0], 4.0, 8.0, 5.4, 0.1) # Longitud del sépalo
sepal_width = st.slider(feature_names[1], 2.0, 4.5, 3.4, 0.1)  # Anchura del sépalo
petal_length = st.slider(feature_names[2], 1.0, 7.0, 1.3, 0.1) # Longitud del pétalo
petal_width = st.slider(feature_names[3], 0.1, 2.5, 0.2, 0.1)  # Anchura del pétalo

# --- 4. Botón para Realizar la Predicción ---
if st.button("Predecir Especie"):
    # Prepara los datos de entrada para el modelo
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=feature_names)

    # Realiza la predicción
    prediction_numeric = model.predict(input_data)[0]
    prediction_species = species_map[prediction_numeric]

    # Muestra el resultado
    st.success(f"La especie de Iris predicha es: **{prediction_species}**")

    # Opcional: Mostrar las probabilidades de cada clase
    prediction_proba = model.predict_proba(input_data)[0]
    st.subheader("Probabilidades de las Clases:")
    proba_df = pd.DataFrame({
        'Especie': list(species_map.values()),
        'Probabilidad': prediction_proba
    }).set_index('Especie')
    st.bar_chart(proba_df)

# --- 5. Información Adicional (Opcional) ---
st.sidebar.header("Acerca de")
st.sidebar.info("Este demo utiliza un modelo RandomForestClassifier entrenado con el famoso dataset Iris.")
st.sidebar.info("Desarrollado con Streamlit y scikit-learn.")
st.sidebar.info("Modelo cargado con `st.cache_resource` para optimizar el rendimiento.")