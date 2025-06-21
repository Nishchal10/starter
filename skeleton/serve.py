from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import traceback

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

class EmailInput(BaseModel):
    text: str

app = FastAPI()

@app.post("/classify")
def classify_email(request: EmailInput):
    try:
        X_input = vectorizer.transform([request.text])

        prediction = model.predict(X_input)[0]

        return {"intent": prediction}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}