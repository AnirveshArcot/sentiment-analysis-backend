from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware
import gdown




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
google_drive_link = 'https://drive.google.com/uc?id=1o7IjKFHX2u8R-tsahSkSp3aEPhBNd1c2'
gdown.download(google_drive_link, output=None, quiet=False)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load("distilbert_sst2.pth",map_location=torch.device('cpu')))
model.eval()
class TextInput(BaseModel):
    text: str
@app.post("/analyze/")
async def analyze_sentiment(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    confidence = probabilities[0][predicted_class].item()

    return {"label": sentiment, "score": confidence}





