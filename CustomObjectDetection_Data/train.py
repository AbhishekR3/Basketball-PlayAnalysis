# Training/Testing the YOLO model on the custom dataset

from ultralytics import YOLO

# Build YOLOv9c model from pre-trained weight
model = YOLO("yolov9c.yaml]")

# Model Information
model.info()

# Train dataset
results = model.train(data="coco8.yaml", epochs=10, imgsz=640)  # train the model


# Use the model

#metrics = model.val()  # evaluate model performance on the validation set