from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from model.predict import MODEL_VERSION, model, artifacts
from schema.user_input import UserInput
from config.feature_validator import feature_col
import pandas as pd
import uvicorn
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, 'static')), name='static')
template = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return template.TemplateResponse(
        request=request,
        name='index.html',
        context={"title": "This is a home page"}
    )

@app.get('/health')
def health_check():
     return{
          "status":"ok",
          "version": MODEL_VERSION,
          "model": model is not None,
          "artifacts": artifacts is not None
     }

@app.post('/submit')
async def form_submit(request: Request):
    form = await request.form()

    data = dict(form)
    print("form submitted")

    try:
        user_input = UserInput(**data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_input=False))

    features = {
        "amt": user_input.amount,
        "category_idx": user_input.cat2id,
        "hour": user_input.hour,
        "day_of_week": user_input.day_of_week,
        "is_weekend": user_input.is_weekend,
        "age": user_input.age,
        "distance_km": user_input.distance_km,
        "city_pop": user_input.city_population,
        "is_online": user_input.is_online,
    }

    X = pd.DataFrame([features], columns=feature_col)

    prediction = model.predict(X)[0]
    print("Input features:", X)
    print("Prediction:", prediction)
    print("Model classes:", model.classes_)

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    print("Prediction:", prediction)
    print("Model classes:", model.classes_)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

        for class_label, probability in zip(model.classes_, probabilities):
            print(f"Class {class_label} probability: {probability:.4%}")
    else:
        print("This model does not support predict_proba().")

    return {
        "message": "Form submitted successfully",
        "features": features,
        "prediction": prediction
        }

if __name__ == "__main__":
    uvicorn.run(reload=True, debug=True)
