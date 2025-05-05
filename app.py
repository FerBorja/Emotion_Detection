import streamlit as st
from transformers import pipeline

# Cambia el modelo a uno específico para emociones
model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

st.title("Detector de Emociones en Texto")
user_input = st.text_input("Escribe algo:")
if user_input:
    result = model(user_input)
    st.write(f"Emoción detectada: {result[0]['label']} (Confianza: {result[0]['score']:.2f})")
