import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Configuración básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ESTILO CORREGIDO: Usamos un estilo disponible
plt.style.use('ggplot')

def load_resources():
    """Carga modelo, tokenizer y datos"""
    try:
        # 1. Cargar datos
        eval_df = pd.read_csv('./cleaned_emotion_dataset.csv')
        
        # 2. Cargar modelo y tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('./emotion_model')
        model = DistilBertForSequenceClassification.from_pretrained('./emotion_model')
        
        # 3. Preparar LabelEncoder
        le = LabelEncoder()
        le.fit(eval_df['emotion'])
        
        # 4. Tokenizar datos
        encodings = tokenizer(
            eval_df['cleaned_text'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # 5. Crear dataset
        class EmotionDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        dataset = EmotionDataset(encodings, le.transform(eval_df['emotion']))
        
        return model, dataset, le, eval_df
    
    except Exception as e:
        logger.error(f"Error cargando recursos: {str(e)}")
        raise

def evaluate_small_dataset(model, dataset, le, df):
    """Evaluación para datasets pequeños (<10 ejemplos)"""
    try:
        trainer = Trainer(model=model)
        predictions = trainer.predict(dataset)
        
        # Convertir predicciones
        pred_probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
        y_pred = np.argmax(pred_probs, axis=1)
        y_true = predictions.label_ids
        
        # Crear figura
        plt.figure(figsize=(10, 6))
        
        # Visualización de probabilidades
        for i, (text, true_emotion, probs) in enumerate(zip(
            df['cleaned_text'],
            le.inverse_transform(y_true),
            pred_probs
        )):
            plt.subplot(len(df), 1, i+1)
            bars = plt.bar(le.classes_, probs, color='skyblue')
            bars[np.argmax(probs)].set_color('salmon')
            plt.title(f"Texto: {text[:50]}...\nVerdadero: {true_emotion}")
            plt.ylim(0, 1)
            if i == len(df)-1:
                plt.xlabel("Emociones")
            plt.ylabel("Probabilidad")
        
        plt.tight_layout()
        
        # Guardar resultados
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/small_dataset_evaluation.png')
        logger.info("Gráfico de probabilidades guardado en ./results/small_dataset_evaluation.png")
        plt.show()
        
        # Mostrar resumen en consola
        logger.info("\n🔍 Análisis por texto:")
        for i, (text, true, pred) in enumerate(zip(
            df['cleaned_text'],
            le.inverse_transform(y_true),
            le.inverse_transform(y_pred)
        )):
            logger.info(f"\n📝 Ejemplo {i+1}:")
            logger.info(f"Texto: {text}")
            logger.info(f"Real: {true} - Predicción: {pred}")
            logger.info("Probabilidades:")
            for emotion, prob in zip(le.classes_, pred_probs[i]):
                logger.info(f"  {emotion}: {prob:.2%}")
        
    except Exception as e:
        logger.error(f"Error en evaluación: {str(e)}")
        raise

def main():
    logger.info("🚀 Iniciando evaluación para dataset pequeño")
    
    # Cargar recursos
    model, dataset, le, df = load_resources()
    
    # Evaluación adaptada al tamaño
    if len(df) < 10:
        logger.warning(f"⚠️ Dataset pequeño detectado ({len(df)} ejemplos). Usando modo de evaluación especial.")
        evaluate_small_dataset(model, dataset, le, df)
    else:
        logger.info("Dataset suficiente. Usando evaluación estándar.")
        # Aquí iría la evaluación normal
    
    logger.info("✅ Evaluación completada")

if __name__ == '__main__':
    main()