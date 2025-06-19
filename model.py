import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path

# Шляхи
DATA_DIR = Path("E:/Магістратура/Практика/data")
MODEL_DIR = Path("E:/Магістратура/Практика/BERT_RAG_medical_texts/model")
CLEANED_MTSAMPLES_PATH = DATA_DIR / "cleaned_mtsamples.csv"

class MedicalDataset(torch.utils.data.Dataset):
    """Клас для підготовки даних для BioBERT."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_biobert():
    """Тренування моделі BioBERT."""
    # Завантаження даних
    df = pd.read_csv(CLEANED_MTSAMPLES_PATH)
    
    # Зменшення датасету до 20% для зменшення ітерацій
    df = df.sample(frac=0.2, random_state=42)
    
    X = df["cleaned_transcription"]
    y = df["label"]

    # Розподіл даних
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Завантаження токенізатора та моделі
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=2)

    # Токенізація
    def tokenize_data(texts, tokenizer, max_length=512):
        return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    train_encodings = tokenize_data(X_train, tokenizer)
    test_encodings = tokenize_data(X_test, tokenizer)

    # Створення датасетів
    train_dataset = MedicalDataset(train_encodings, y_train.values)
    test_dataset = MedicalDataset(test_encodings, y_test.values)

    # Налаштування параметрів тренування
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        save_safetensors=False,  # Вимкнення safetensors
    )

    # Кастомний Trainer для забезпечення континуальності тензорів
    class CustomTrainer(Trainer):
        def __init__(self, *args, tokenizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer = tokenizer  # Зберігаємо токенізатор

        def save_model(self, output_dir=None, _internal_call=False):
            if output_dir is None:
                output_dir = self.args.output_dir
            self.model.eval()
            for param in self.model.parameters():
                param.data = param.data.contiguous()  # Забезпечення континуальності
            self.model.save_pretrained(output_dir, safe_serialization=False)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)  # Збереження токенізатора

    # Тренування моделі
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,  # Передаємо токенізатор
    )
    trainer.train()

    # Оцінка моделі
    predictions = trainer.predict(test_dataset)
    y_pred = predictions.predictions.argmax(-1)
    print("F1-score:", f1_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Збереження фінальної моделі
    for param in model.parameters():
        param.data = param.data.contiguous()
    model.save_pretrained(MODEL_DIR, safe_serialization=False)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")

def predict_diagnosis(text, model_path=str(MODEL_DIR)):
    """Передбачення діагнозу для введеного тексту."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "Acute" if prediction == 0 else "Chronic"

if __name__ == "__main__":
    train_biobert()