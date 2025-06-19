import pandas as pd
import spacy
from pathlib import Path

# Налаштування spaCy
nlp = spacy.load("en_core_web_sm")

# Шляхи до файлів
DATA_DIR = Path("E:/Магістратура/Практика/data")
MTSAMPLES_PATH = DATA_DIR / "mtsamples.csv"
PUBMED_DIR = DATA_DIR / "PubMed_20k_RCT"
CLEANED_MTSAMPLES_PATH = DATA_DIR / "cleaned_mtsamples.csv"
CLEANED_PUBMED_PATH = DATA_DIR / "cleaned_pubmed.csv"

def preprocess_text(text):
    """Очищення тексту: токенізація, лематизація, видалення стоп-слів."""
    if pd.isna(text):
        return ""
    doc = nlp(str(text))
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def prepare_mtsamples():
    """Очищення та підготовка датасету Medical Transcriptions."""
    # Завантаження датасету
    df = pd.read_csv(MTSAMPLES_PATH)
    
    # Видалення пропущених значень
    df = df.dropna(subset=["transcription", "keywords"])
    
    # Очищення тексту
    df["cleaned_transcription"] = df["transcription"].apply(preprocess_text)
    
    # Створення міток (приклад: на основі keywords)
    df["label"] = df["keywords"].apply(lambda x: 0 if "acute" in x.lower() else 1)  # 0: Acute, 1: Chronic
    
    # Збереження очищеного датасету
    df.to_csv(CLEANED_MTSAMPLES_PATH, index=False)
    return df

def prepare_pubmed():
    """Очищення та підготовка датасету PubMed 20k RCT."""
    # Завантаження всіх файлів
    dfs = []
    for split in ["train.csv", "dev.csv", "test.csv"]:
        df = pd.read_csv(PUBMED_DIR / split)
        dfs.append(df)
    
    # Об'єднання даних
    pubmed_df = pd.concat(dfs, ignore_index=True)
    
    # Використання колонки 'abstract_text' замість 'abstract'
    pubmed_df["cleaned_abstract"] = pubmed_df["abstract_text"].apply(preprocess_text)
    
    # Видалення пропущених значень
    pubmed_df = pubmed_df.dropna(subset=["cleaned_abstract"])
    
    # Збереження очищеного датасету
    pubmed_df.to_csv(CLEANED_PUBMED_PATH, index=False)
    return pubmed_df

if __name__ == "__main__":
    print("Processing Medical Transcriptions...")
    mtsamples_df = prepare_mtsamples()
    print(f"Cleaned Medical Transcriptions saved to {CLEANED_MTSAMPLES_PATH}")
    
    print("Processing PubMed 20k RCT...")
    pubmed_df = prepare_pubmed()
    print(f"Cleaned PubMed 20k RCT saved to {CLEANED_PUBMED_PATH}")