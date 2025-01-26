import os
from src.train import train_model

def test_training_script():
    # Run the training function
    train_model()
    
    # Check if the model file is created
    assert os.path.exists("model.pkl")
