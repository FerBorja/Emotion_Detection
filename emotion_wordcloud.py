import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv("cleaned_emotion_dataset.csv")

# Selecciona la emoción
emotion = "joy"  # Cambia según la emoción

# Genera el texto concatenado para esa emoción
text = " ".join(df[df["emotion"] == emotion]["cleaned_text"])

# Crear la nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

# Mostrar la nube
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title(f"Palabras más frecuentes en '{emotion}'")
plt.show()
