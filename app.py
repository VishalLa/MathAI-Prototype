from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import torch
from Utils.pre_process import preprocess_canvas
from Covonutional_neural_network.network import CNN
from Covonutional_neural_network.modelUttils.model_utils import load_model

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
MODEL_PATH = "C:\\Users\\visha\\OneDrive\\Desktop\\New folder\\Model\\model_parameters.pth"
model = CNN()
model = load_model(model, MODEL_PATH)

# Define the request schema
class PredictionRequest(BaseModel):
    actions: list  # List of actions (2D array representing the canvas)

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

        # Preprocess the actions array (convert to tensor, etc.)
        input_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logging.info(f"Input tensor shape: {input_tensor.shape}")

        # Make a prediction
        predicted_class = preprocess_canvas(input_tensor)

        logging.info(f"Prediction: {predicted_class}")
        return {"prediction": predicted_class}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
