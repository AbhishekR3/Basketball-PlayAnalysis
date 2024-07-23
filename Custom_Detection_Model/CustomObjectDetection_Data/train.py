'''
Training the YOLO model on the custom dataset

Key Concepts Implemented:
- Augmentations (refer README.dataset.txt for more info)
- Logging GPU memory usage
'''

# Training/Testing the YOLO model on the custom dataset

# Import Libraries
import torch
from ultralytics import YOLO
import os
import numpy as np
import random
from sklearn.model_selection import KFold

# Check Pytorch version and ensure GPU is being used on device (macOS users)
print(f"PyTorch version: {torch.__version__}")
print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
print(f"MPS backend enabled: {torch.backends.mps.is_built()}")

# Set seeds for consistentncy

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
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

# Build YOLOv9c model from pre-trained weight
model = YOLO("yolov9c.pt")

# Move model to appropriate device
model.to(device)

# Model Information
model.info()

# GPU memory usage before training
print()
print("Memory usage before training:")
log_memory_usage()
print()

# Train dataset
projectfile_path = os.getcwd()
trainingdata_path = os.path.join(projectfile_path, 'Custom_Detection_Model', 'CustomObjectDetection_Data', 'data.yaml')
results = model.train(data=trainingdata_path, epochs=10, imgsz=640, device=device)  # train the model

# GPU memory usage after training
print()
print("Memory usage after training:")
log_memory_usage()
print()

# Use the model
metrics = model.val()  # evaluate model performance on the validation set
print(metrics)

# Save the model
model_path = os.path.join(projectfile_path, 'CustomModel_InstanceSegmentation.pt')
model.save(model_path)