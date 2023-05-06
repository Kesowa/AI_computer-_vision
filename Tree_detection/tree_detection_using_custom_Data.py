import torch
import time
import os
from deepforest import main

class DeepForestModel:
    def __init__(self):
        # Instantiate a new model object
        self.model = main.deepforest()
        self.model.use_release()
        
    def load_model(self, model_path):
        # Load the saved state dictionary into the new model object
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
    
    def train(self, train_data):
        start_time = time.time()
        self.model.trainer.fit(train_data)
        print(f"--- Training on CPU: {(time.time() - start_time):.2f} seconds ---")
        
    def evaluate(self, test_file, save_dir):
        results = self.model.evaluate(test_file, os.path.dirname(test_file), iou_threshold=0.4, savedir=save_dir)
        return results
        
        
# Create an instance of the class
my_model = DeepForestModel()

# Load the saved model
model_path = "my_model.pt"
my_model.load_model(model_path)


# Evaluate the model on some test data
test_file = r'C:\Users\FS-AI\Downloads\Tree canopy\Chayan_mandal\train_example.csv'
save_dir = r'C:\save_m'
results = my_model.evaluate(test_file, save_dir)
