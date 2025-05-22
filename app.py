from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import torch
from CNN_Model.Utils.pre_process import prepare_canvas, predict_charheacters
from CNN_Model.Covonutional_neural_network.CNNnetwork import CNN
from CNN_Model.Covonutional_neural_network.modelUttils.model_utils import load_model

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Load the trained model
MODEL_PATH = "C:\\Users\\visha\\OneDrive\\Desktop\\MathAI\\CNN_Model\\model_parameters.pth"
network = CNN()
model = load_model(network, MODEL_PATH)

class PredictionRequest(BaseModel):
    actions: list

@app.get("/")
def home():
    """Default route to confirm the server is running."""
    return {"message": "Model API is running!"}

@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    """Endpoint to handle predictions."""
    try:
        actions = request.actions
        logging.info(f"Received actions: {actions}")

        if not actions:
            raise HTTPException(status_code=400, detail="No actions provided")

        processed_canvas = prepare_canvas(actions)
        logging.info(f"Processed canvas shape: {processed_canvas.shape}")

        predicted_class = predict_charheacters(model=model, canvas_array=processed_canvas)

        logging.info(f"Prediction: {predicted_class}")
        return {"prediction": predicted_class}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
