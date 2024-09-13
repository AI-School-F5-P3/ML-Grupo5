import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import csv
from datetime import datetime
import os
from src.database.connection import create_connection, close_connection
import numpy as np

# Configuración para suprimir advertencias de TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Función para cargar modelos de manera segura
def load_model(file_path):
    try:
        if file_path.endswith('.keras'):
            return tf.keras.models.load_model(file_path)
        model = joblib.load(file_path)
        if isinstance(model, dict) and 'model' in model:
            return model['model']
        return model
    except FileNotFoundError:
        st.error(f"No se pudo encontrar el archivo del modelo: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo {file_path}: {str(e)}")
        return None

# Cargar los modelos
model_dir = 'src/Modelos'  # Ajusta esta ruta según la estructura de tu proyecto
xgboost_model = load_model(os.path.join(model_dir, 'xgboost_model.joblib'))
logistic_model = load_model(os.path.join(model_dir, 'logistic_model.joblib'))
stack_model = load_model(os.path.join(model_dir, 'stack_model.joblib'))
neuronal_model = load_model(os.path.join(model_dir, 'neuronal.keras'))

# Función para hacer predicciones
def predict(data, model):
    if model is not None:
        try:
            if isinstance(model, tf.keras.Model):
                pred = model.predict(data)
                return pred[0][0] > 0.5
            else:
                pred = model.predict(data)[0]
            return pred
        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")
    return None

# Función para guardar los resultados en CSV
def save_to_csv(data, predictions, feedback):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = 'src/Data/feedback_results.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Timestamp'] + list(data.keys()) + 
                            ['XGBoost_pred', 'XGBoost_fback', 'LogisticRegression_pred', 'LogisticRegression_fback',
                            'StackModel_pred', 'StackModel_fback', 'RedNeuronal_pred', 'RedNeuronal_fback'])
        writer.writerow([timestamp] + list(data.values()) + [
            predictions['Modelo1'],
            feedback['Modelo1'],
            predictions['Modelo2'],
            feedback['Modelo2'],
            predictions['Modelo3'],
            feedback['Modelo3'],
            predictions['Modelo4'],
            feedback['Modelo4']
        ])

def save_to_database(data, predictions, feedback):
    connection = create_connection()
    cursor = connection.cursor()

    # Convert numpy types to Python native types
    def convert_to_python_type(value):
        if isinstance(value, (np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        return value

    # Convert all values in data, predictions, and feedback
    data = {k: convert_to_python_type(v) for k, v in data.items()}
    predictions = {k: convert_to_python_type(v) for k, v in predictions.items()}
    feedback = {k: convert_to_python_type(v) for k, v in feedback.items()}

    # Prepare the query
    query = """
    INSERT INTO new_data (
        gender, customer_type, age, travel_type, flight_class, flight_distance,
        inflight_wifi, departure_convenience, online_booking, gate_location,
        food_drink, online_boarding, seat_comfort, inflight_entertainment, onboard_service,
        legroom_service, baggage_handling, checkin_service, inflight_service_personal,
        cleanliness, departure_delay, arrival_delay,
        xgboost_prediction, logistic_prediction, stacked_prediction, neural_prediction,
        feedback_model1, feedback_model2, feedback_model3, feedback_model4
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Prepare the values
    values = (
        data['Gender'], data['Customer Type'], data['Age'], data['Type of Travel'], data['Class'], data['Flight Distance'],
        data['Inflight wifi service'], data['Departure/Arrival time convenient'], data['Ease of Online booking'], data['Gate location'],
        data['Food and drink'], data['Online boarding'], data['Seat comfort'], data['Inflight entertainment'], data['On-board service'],
        data['Leg room service'], data['Baggage handling'], data['Checkin service'], data['Inflight service'],
        data['Cleanliness'], data['Departure Delay in Minutes'], data['Arrival Delay in Minutes'],
        predictions['Modelo1'], predictions['Modelo2'], predictions['Modelo3'], predictions['Modelo4'],
        feedback['Modelo1'], feedback['Modelo2'], feedback['Modelo3'], feedback['Modelo4']
    )

    cursor.execute(query, values)
    connection.commit()
    close_connection(connection)

# Crear la aplicación Streamlit
st.title('Encuesta de Satisfacción')

# Bienvenida personalizada
st.markdown("""
**¡Gracias por volar con nosotros!** 
Te invitamos a compartir tu experiencia en esta breve encuesta (menos de 2 minutos). Tu opinión es muy valiosa para mejorar nuestros servicios.
""")

# Crear el formulario
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

# Inputs de selección en dos columnas
with col1:
    gender = st.selectbox('Género', ['Masculino', 'Femenino'])
    customer_type = st.selectbox('Tipo de Cliente', ['Leal', 'Desleal'])
    age = st.number_input('Edad', min_value=0, max_value=120)
with col2:
    type_of_travel = st.selectbox('Tipo de Viaje', ['Personal', 'Negocios'])
    class_type = st.selectbox('Clase', ['Económica', 'Económica Plus', 'Negocios'])
    flight_distance = st.number_input('Distancia de Vuelo', min_value=0)

def satisfaction_radio(label, var_name):
    emojis = ['😠', '😞', '😐', '😊', '😁']  # Emojis de 0 a 5
    st.write(label)
    selected_value = st.radio(
        label,
        options=emojis,
        index=0,
        key=var_name,
        horizontal=True
    )
    return emojis.index(selected_value)+1

# Columnas de satisfacción
inflight_wifi_service = satisfaction_radio('Servicio de WiFi a bordo', 'inflight_wifi_service')
departure_arrival_time_convenient = satisfaction_radio('Conveniencia del horario de salida/llegada', 'departure_arrival_time_convenient')
ease_of_online_booking = satisfaction_radio('Facilidad de reserva en línea', 'ease_of_online_booking')
gate_location = satisfaction_radio('Ubicación de la puerta', 'gate_location')
food_and_drink = satisfaction_radio('Comida y bebida', 'food_and_drink')
online_boarding = satisfaction_radio('Embarque en línea', 'online_boarding')
seat_comfort = satisfaction_radio('Confort del asiento', 'seat_comfort')
inflight_entertainment = satisfaction_radio('Entretenimiento a bordo', 'inflight_entertainment')
onboard_service = satisfaction_radio('Servicio a bordo', 'onboard_service')
leg_room_service = satisfaction_radio('Espacio para las piernas', 'leg_room_service')
baggage_handling = satisfaction_radio('Manejo de equipaje', 'baggage_handling')
checkin_service = satisfaction_radio('Servicio de check-in', 'checkin_service')
inflight_service = satisfaction_radio('Servicio durante el vuelo', 'inflight_service')
cleanliness = satisfaction_radio('Limpieza', 'cleanliness')

# Inputs numéricos en dos columnas
with col5:
    departure_delay = st.number_input('Retraso en la salida (minutos)', min_value=0)

with col6:
    arrival_delay = st.number_input('Retraso en la llegada (minutos)', min_value=0)

# Inicializar variables de estado
if 'results_shown' not in st.session_state:
    st.session_state.results_shown = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Botón para hacer la predicción
if st.button('Ver Resultados'):
    # Preparar los datos para la predicción
    st.session_state.data = {
        'Gender': 0 if gender == 'Masculino' else 1,
        'Customer Type': 0 if customer_type == 'Leal' else 1,
        'Age': age,
        'Type of Travel': 0 if type_of_travel == 'Personal' else 1,
        'Class': 0 if class_type == 'Económica' else (1 if class_type == 'Económica Plus' else 2),
        'Flight Distance': flight_distance,
        'Inflight wifi service': inflight_wifi_service,
        'Departure/Arrival time convenient': departure_arrival_time_convenient,
        'Ease of Online booking': ease_of_online_booking,
        'Gate location': gate_location,
        'Food and drink': food_and_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': inflight_entertainment,
        'On-board service': onboard_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Departure Delay in Minutes': departure_delay,
        'Arrival Delay in Minutes': arrival_delay,
    }
    
    # Convertir el diccionario de datos en un DataFrame
    data_df = pd.DataFrame([st.session_state.data])
    
    # Hacer predicciones
    st.session_state.predictions = {
        'Modelo1': predict(data_df, xgboost_model),
        'Modelo2': predict(data_df, logistic_model),
        'Modelo3': predict(data_df, stack_model),
        'Modelo4': predict(data_df, neuronal_model)
    }
    
    st.session_state.results_shown = True

# Mostrar resultados si están disponibles
if st.session_state.results_shown:
    st.subheader("Resultados de las predicciones")
    
    # Crear un DataFrame con los resultados y la columna de feedback
    results_df = pd.DataFrame({
        'Modelo': ['Modelo1', 'Modelo2', 'Modelo3', 'Modelo4'],
        'Predicción': [('Satisfecho' if pred == 1 else 'Insatisfecho') for pred in st.session_state.predictions.values()],
        'Feedback': ['Sí'] * 4  # Inicializar con 'Sí' en lugar de cadenas vacías
    })

    # Usar st.data_editor para mostrar los resultados y permitir la edición del feedback
    edited_df = st.data_editor(
        results_df,
        column_config={
            "Modelo": st.column_config.Column(
                width="medium",
                help="Nombre del modelo utilizado para la predicción",
            ),
            "Predicción": st.column_config.Column(
                width="medium",
            ),
            "Feedback": st.column_config.SelectboxColumn(
                width="medium",
                options=["Sí", "No"],
                required=True,
                help="¿Estás de acuerdo con la predicción?",
            )
        },
        disabled=["Modelo", "Predicción"],
        hide_index=True,
    )

    # Botón para enviar el formulario
    if st.button('Enviar Formulario'):
        # Verificar si todos los feedbacks han sido proporcionados
        if edited_df['Feedback'].isnull().sum() == 0:
            # Codificar el feedback: Sí -> 1, No -> 0
            feedback = dict(zip(edited_df['Modelo'], (edited_df['Feedback'] == 'Sí').astype(int)))
            
            # Guardar en CSV
            save_to_csv(st.session_state.data, st.session_state.predictions, feedback)        

            # Guardar en base de datos
            save_to_database(st.session_state.data, st.session_state.predictions, feedback)
            
            # Reiniciar las variables de estado
            st.session_state.results_shown = False
            st.session_state.data = None
            st.session_state.predictions = None

            # Mensaje de agradecimiento
            st.success('¡Muchas gracias por tu colaboración! Tu opinión nos ayudará a mejorar tu próxima experiencia de vuelo.')
        else:
            st.warning('Por favor, proporciona feedback para todas las predicciones antes de enviar el formulario.')
