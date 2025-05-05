import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

# 1. Cargar el dataset (usando el archivo local que generó el primer script)
input_path = os.path.join(os.path.dirname(__file__), 'local_emotion_dataset.csv')

try:
    df = pd.read_csv(input_path)
    print(f"Dataset cargado desde: {input_path}")
    print(f"Registros cargados: {len(df)}")
    
except FileNotFoundError:
    print("\n⚠️ Error: No se encontró el dataset local")
    print("Ejecuta primero dataset.py para generarlo")
    exit()

# 2. Configurar NLTK
nltk.download('stopwords', quiet=True)  # Descarga silenciosa si no existe

# 3. Función de limpieza mejorada
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)  # Eliminar URLs/mentions
    text = re.sub(r"[^\w\s]", "", text.lower())            # Eliminar puntuación
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# 4. Aplicar limpieza (usando la columna correcta)
# Verifica si existe la columna 'text' o 'content'
text_column = 'text' if 'text' in df.columns else 'content'
df["cleaned_text"] = df[text_column].apply(clean_text)

# 5. Mostrar resultados
print("\nAntes y después de la limpieza:")
print(df[[text_column, "cleaned_text"]].head(10))

# 6. Guardar resultados
output_path = os.path.join(os.path.dirname(__file__), 'cleaned_emotion_dataset.csv')
df.to_csv(output_path, index=False)
print(f"\nDataset limpio guardado en: {output_path}")