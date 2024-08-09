'''
Training/Testing the YOLO model on the custom dataset

Key Concepts Implemented:
- Augmentations (refer README.dataset.txt for more info)
- Enabled GPU-accelerated programming
- Early Stopping + Cosine Learning Rate for model training
'''

# Import Libraries
import torch
from ultralytics import YOLO
import os
import numpy as np
import random

# Check Pytorch version and ensure GPU is being used on device (macOS users)
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
print(f"MPS backend enabled: {torch.backends.mps.is_built()}")

# Set seeds for consistentncy

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds()
      
# Check GPU memory usage
def log_memory_usage():
    if torch.backends.mps.is_available():
        print(f"Memory allocated: {torch.mps.current_allocated_memory() / 1e6:.2f} MB")
        print(f"Memory reserved: {torch.mps.driver_allocated_memory() / 1e6:.2f} MB")


# Check if GPU (MPS) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

# Build YOLO model from pre-trained weight
YOLO_pretrained_model_path = 'Custom_Detection_Model/CustomObjectDetection_Data/yolov10s.pt'
model = YOLO(YOLO_pretrained_model_path)

# Move model to appropriate device
model.to(device)

# Model Information
model.info()

# GPU memory usage before training
print()
print("Memory usage before training:")
log_memory_usage()
print()

# Train dataset with parameters 
projectfile_path = os.getcwd()
trainingdata_path = os.path.join(projectfile_path, 'Custom_Detection_Model', 'CustomObjectDetection_Data', 'data.yaml')
#trainingdata_path = os.path.join(projectfile_path, 'CustomObjectDetection_Data-1', 'data.yaml')

results = model.train(
    data=trainingdata_path, 
    epochs=15, 
    imgsz=640, 
    device=device,
    lr0=0.01,
    lrf=0.05,
    cos_lr=True,  # Cosine Learning rate
    patience=4,  # Early Stopping
    save_period=1,
    verbose=True)

# GPU memory usage after training
print()
print("Memory usage after training:")
log_memory_usage()
print()

# Use the model
print('Metrics:')
metrics = model.val()  # evaluate model performance on the validation set
print(metrics)

'''
#%%
from roboflow import Roboflow
rf = Roboflow(api_key="MzkEnQgS74xzyQ2z92O8")
project = rf.workspace("basketballplayanalysis").project("customobjectdetection_data")
version = project.version(4)
dataset = version.download("yolov5")
'''