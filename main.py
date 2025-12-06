from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "tasha-raah/roberta-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

model.config.id2label = {
    0: "NEGATIVE",
    1: "POSITIVE",
}
model.config.label2id = {
    "NEGATIVE": 0,
    "POSITIVE": 1,
}

model.eval()

app = FastAPI(title="RoBERTa Sentiment API")

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=SentimentResponse)
def predict(req: SentimentRequest):
    inputs = tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    score, pred_id = torch.max(probs, dim=0)

    # Use our custom mapping
    id2label = model.config.id2label
    label = id2label[int(pred_id)]

    return SentimentResponse(label=label, score=float(score))
