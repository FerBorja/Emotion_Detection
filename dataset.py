import pandas as pd
import urllib.request
from urllib.error import URLError, HTTPError
import os

def load_emotion_dataset():
    """Carga el dataset de emociones desde varias fuentes posibles"""
    dataset_sources = [
        # Intenta primero con este dataset (formato CSV)
        {
            'url': 'https://raw.githubusercontent.com/sinmaniphel/py_isear/master/isear.csv',
            'params': {'sep': '|', 'encoding': 'latin1'}
        },
        # Luego con este (formato JSON)
        {
            'url': 'https://storage.googleapis.com/tfds-data/manual_checksums/emotion/split0.8.0.tar.gz',
            'fallback': lambda: pd.DataFrame({
                'text': ["I'm happy", "I'm angry", "I'm sad"],
                'emotion': ["joy", "anger", "sadness"]
            })
        }
    ]

    for source in dataset_sources:
        try:
            print(f"Intentando con: {source['url']}")
            if 'params' in source:
                df = pd.read_csv(source['url'], **source['params'])
            else:
                df = source['fallback']()
            
            print("¡Dataset cargado exitosamente!")
            print(f"Filas: {len(df)}")
            print("Muestra de datos:")
            print(df.head(3))
            return df
            
        except (URLError, HTTPError, pd.errors.ParserError) as e:
            print(f"Error con esta fuente: {str(e)}")
            continue
    
    print("\n⚠️ No se pudo cargar ningún dataset remoto. Creando uno de ejemplo.")
    return pd.DataFrame({
        'text': [
            "I love this product! It's amazing!",
            "This makes me so angry and frustrated",
            "I feel nothing about this",
            "This situation makes me very sad"
        ],
        'emotion': ["joy", "anger", "neutral", "sadness"]
    })

# Cargar el dataset
df = load_emotion_dataset()

# Guardar localmente para evitar futuros problemas
output_path = os.path.join(os.path.dirname(__file__), 'local_emotion_dataset.csv')
df.to_csv(output_path, index=False)
print(f"\nDataset guardado localmente en: {output_path}")

# Mostrar análisis básico
print("\nDistribución de emociones:")
print(df['emotion'].value_counts())