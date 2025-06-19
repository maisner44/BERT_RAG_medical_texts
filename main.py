from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_diagnosis
from pathlib import Path

app = FastAPI()

# Модель для вхідних даних
class TextInput(BaseModel):
    text: str

# Шлях до моделі
MODEL_PATH = Path("E:/Магістратура/Практика/BERT_RAG_medical_texts/model")

@app.post("/classify")
async def classify_diagnosis(input: TextInput):
    """Ендпоінт для класифікації діагнозу."""
    diagnosis = predict_diagnosis(input.text, model_path=str(MODEL_PATH))
    return {"diagnosis": diagnosis}

@app.post("/search_articles")
async def search_articles(input: TextInput):
    """Ендпоінт-заглушка для пошуку статей (RAG)."""
    return {"message": "RAG pipeline will be implemented later", "input_text": input.text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)