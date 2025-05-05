import logging
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from urllib.error import URLError, HTTPError

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_emotion_dataset():
    dataset_sources = [
        {
            'url': 'https://raw.githubusercontent.com/sinmaniphel/py_isear/master/isear.csv',
            'params': {'sep': '|', 'encoding': 'latin1'}
        },
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
            logger.info(f"🌐 Intentando cargar: {source['url']}")
            if 'params' in source:
                df = pd.read_csv(source['url'], **source['params'])
            else:
                df = source['fallback']()
            df = df.rename(columns=lambda x: x.lower())
            if 'text' not in df.columns or 'emotion' not in df.columns:
                raise ValueError("❌ Las columnas necesarias no están presentes (text, emotion)")
            return df[['text', 'emotion']]
        except (URLError, HTTPError, pd.errors.ParserError, ValueError) as e:
            logger.warning(f"⚠️ Error con esta fuente: {e}")
            continue

    logger.warning("⚠️ No se pudo cargar ningún dataset remoto. Usando ejemplo.")
    return pd.DataFrame({
        'text': [
            "I love this product! It's amazing!",
            "This makes me so angry and frustrated",
            "I feel nothing about this",
            "This situation makes me very sad"
        ],
        'emotion': ["joy", "anger", "neutral", "sadness"]
    })

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def main():
    try:
        logger.info("📥 Cargando dataset...")
        df = load_emotion_dataset()
        logger.info(f"✅ Datos cargados. Total ejemplos: {len(df)}")

        logger.info("🧹 Limpiando texto...")
        df["cleaned_text"] = df["text"].astype(str).str.strip()

        logger.info("🔡 Tokenizando texto...")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(
            df["cleaned_text"].tolist(),
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        logger.info("🏷️ Codificando etiquetas...")
        le = LabelEncoder()
        labels = le.fit_transform(df["emotion"])
        logger.info(f"🧠 Clases: {list(le.classes_)}")

        dataset = EmotionDataset(inputs, labels)
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(le.classes_)
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="no"
        )

        logger.info("🚀 Iniciando entrenamiento...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        logger.info("🎉 Entrenamiento completado")

        logger.info("💾 Guardando modelo y tokenizer...")
        os.makedirs("./emotion_model", exist_ok=True)
        model.save_pretrained("./emotion_model")
        tokenizer.save_pretrained("./emotion_model")
        logger.info("✅ Modelo guardado en './emotion_model'")

    except Exception as e:
        logger.error(f"⛔ Error inesperado: {e}")

if __name__ == "__main__":
    main()
