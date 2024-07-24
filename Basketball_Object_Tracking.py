'''
Basketball Object Tracking
This file tracks the positions of each player and the basketball.

Key Concepts Implemented:
- Hough Circle Transform - Object Detection specifically for Circles
- YOLO - End to End Object Object Detection using YOLOv9c base for accuracy/speed balance
--> Implemented a custom model with 98.4% mAP50 (Refer CustomObjectDetection_Data/README.dataset.txt for more info)
- DeepSort - Multi Object Tracking Algorithm that handles well with occlusion
--> Implemented validation

'''

'''
Upcoming Implementations:

0. 
Create validation test cases for object detection / deepsort tracking

Sequences: Passing, Dribbling
Sequence Level Splitting keep temporal continuity
- 30fps: 1-22: Train, 22-27: Test, 27-30: Validation

Metrics to track
- MOTA
--> This is considers 3 major error ratios
--> MOTA = 1 - (FalseNegatives + FalsePositives + IDSwitches) / GroundTruth

- MOTP 
--> This measures the preecision of object localization
--> MOTP = Σ(Distance_between_objects) / Σ(count_matches)

- IDF1
--> This is the ratio between correctly identified detections and average ground-truth+computed detections
--> IDF1 = 2 * ID_TruePositive /  (2 * ID_TruePositive + ID_FalseNegative + ID_FalsePositive)

- Track Completness
--> This measures the completeness of length of trajectory

- Intersection over Union (IoU)
- Center error
- Track completeness
- ID switches
- False positives/negatives




- Sequences of different lengths

Challenging Situations
- Occlusion

Augmentations needs to be applied to the entire video not varying per frame

K-fold cross-validation


If validation test good, continue with integrating current_time as a feature for the object detection then Cloud Migration.
If validation test not good, fix it with the below in mind
---------
0. Change from mars128 feature extractor to custom YOLO model
<PLANNING>
1. Remove the mars128 feature extractor.
2. Modify the YOLO model to output features along with detections.
3. Update the `object_tracking` function to use YOLO features instead of mars128 features.
4. Adjust the DeepSORT tracker to work with the new feature format.
5. Test the changes and fine-tune as needed.
</PLANNING>

1. Fix the movement of the basketball 
- when receiving a pass
- 

1.
Improve custom model implement: K-Fold Cross Validation?

2.
Feature Extractor Model
- ResNet50
- EfficientNet

3.
Include these additional features: 
- velocity
- acceleration
- color historgram

4.
YOLOv10 implementation

5.
Alpha Blending in game simulation
- Alpha Blending - Rendering semi-transparent objects when there is overlap to increase object detection/tracking

-1. Image segmentation
-2. Perform edge detection to strengthen outline of the circle
-3. Alpha Blending


6.
Optimize parameters

Grid Search
- Kalman filter parameters
- max_dist
- NMS threshold? - Removed in YOLOv10
- other parameters

Include Kalman filter state as a feature for object tracking
'''

#%%

"Import Libraries"

import cv2
import numpy as np
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# DeepSORT code from local files
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

""""
# Set seeds for consistentncy
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use this function before training
set_seeds()
"""
#%%

def preprocess_frame(frame, greyed = True, blur = 'median'):
    """
    Objective:
    Preprocess frame of the video based on input parameters

    Parameters:
    [array]  frame - video frame
    [bool]   greyed (Default: True) - apply a grey filter on the image
    [string] blur (Default: 'median') - apply either median/gaussian blur

    Returns:
    [array] frame - video frame
    """
    try:
        # Convert the image to grey scale for better OpenCV processing
        if greyed is True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        if blur == 'median':
            frame = cv2.medianBlur(frame, 5)
        elif blur == 'gaussian':
            frame = cv2.GaussianBlur(frame, (5,5), 1.5)

        return frame

    except Exception as e:
        logger.error("Error in preprocessing the frame: %s", e)

#%%

def object_tracking(frame, model, tracker, encoder, n_tracked, detected_objects):
    """
    Objective:
    Perform deepsort object tracking on each video frame.
    
    Parameters:
    [array] frame - video frame before object tracking
    [class model] model - YOLO detection with the custom model
    [class deepsort] tracker - DeepSORT Tracker
    [function] encoder - Extracts relevant information (features) from the given frame 
    [int] n_tracked - number of objects that are tracked (debugging purposes)
    [dataframe] detected_objects - pandas dataframe to store information on detected objects

    Returns:
    [array] frame - video frame after object tracking
    [int] n_tracked - number of objects that are tracked (debugging purposes)
    [dataframe] detected_objects - pandas dataframe to store information on detected objects
    """
    
    """
    #Relevant parameters outside of object_tracking() which influences the accuracy of object tracking 
    tracker.py - max_iou_distance=0.5, max_age=2, n_init=5
    B_O_T.py - scores>0.65, max_cosine_distance=0.3, nn_metric=cosine
    --> model=YOLOv9c based custom model

    Check these
    Kalman filter parameters
    max_dist
    NMS threshold
    """

    # Process the current frame with the YOLO model
    results = model(frame)

    # Extract bounding boxes and scores
    boxes = results[0].boxes.xyxy.numpy()
    scores = results[0].boxes.conf.numpy()

    # Filter the detections based on confidence threshold
    mask = scores > 0.65
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Add each tracked object's data in the detected_objects dataframe
    tracked_objects = []
    for i in range(len(boxes)):
        ith_object_details = [
            boxes[i], #BoundingBox
            scores[i], #ConfidenceScore
            np.float32(n_frames/30) #Time
        ]
        new_row = pd.DataFrame([ith_object_details], columns=['BoundingBox', 'ConfidenceScore', 'Time'])
        detected_objects = pd.concat([detected_objects, new_row], ignore_index=True)

    # Compute features for Deep SORT
    features = encoder(frame, boxes)

    # Create detections for Deep SORT
    detections = []

    for box, score, feature in zip(boxes, scores, features):
        # Create a new Detection object
            detection = Detection(
                box, 
                score, 
                feature)
            detections.append(detection)
    
    if detections is None:
        print('No circle features were detected in the frame')
        logger.debug("No circle features were detected in the frame: %s", e)

    # Update tracker
    tracker.predict()
    tracker.update(detections)

    # Calculate number of objects tracked
    print('Number objects tracked:', len(tracker.tracks))
    n_tracked += len(tracker.tracks)

    # Draw bounding boxes and IDs
    for ith_value, track in enumerate(tracker.tracks):
        # Check if
        # not track.is_confirmed()    : Check that an object track has not been found
        # track.time_since_update > 1 : Check the track has not been updated for more than one frame
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # Calculate the coordinates' border of the object
        bbox = track.to_tlbr()

        # Create the borders for the detected objects + relevant track_id and confidence score
        try:
            # Calculate the object's detection confidence score
            confidence_score = scores[ith_value]
        
        except:
            # Calculate the object's detection confidence score
            confidence_score = 0.0

        # Label detected object's track_id and confidence score
        color = (255, 255, 255)  # BGR format

        cv2.putText(frame, f"{track.track_id}-{confidence_score:.3f}", (int(bbox[0]), int(bbox[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, n_tracked, detected_objects

#%% Configuring logging

try:
    log_file_path = 'object_tracking_output.log'

    # If the log file exists, delete it
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        print(f"The file {log_file_path} has been found thus deleted for the new run.")

    # Create a logger object
    logger = logging.getLogger('ObjectTrackingLogger')
    logger.setLevel(logging.DEBUG)  # Set the minimum log level to debug

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

except Exception as e:
    logger.error("Error in creating logging configuration: %s", e)


#%% Initialize Simulation Variables

# Path to the video file / basketball court diagram
script_directory = os.getcwd()
video_path = os.path.join(script_directory, 'assets/Simulation_Cropped.mp4')
basketball_court_diagram = os.path.join(script_directory, 'assets/Basketball Court Diagram.jpg')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Create a mask to remove basketball court diagram from the video
try:
    mask = cv2.imread(basketball_court_diagram, cv2.IMREAD_GRAYSCALE)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask = cv2.resize(mask, (width, height))
    _, mask = cv2.threshold(mask, 255, 255, cv2.THRESH_BINARY) # First number represents the threshold level of removal of the masked image
    mask = mask.astype(np.uint8)

except Exception as e:
    logger.error (f"Error: Couldn't open the basketball court diagram file. {e}")
    exit()

# Parameter values to test
param1_value = 10 # 12/13 - Best results
param2_value = 15 # 15 - Best results

# Initialize results dictionary
resulting_values = {}
resulting_values[(param1_value, param2_value)] = 0

n_frames = 0 # Initialize n_frames to count the number of frames in the video

# Check if the video file opened successfully
if not cap.isOpened():
    logger.error ("Error: Couldn't open the video file.")
    exit()

# Create a VideoWriter object to save the output video
output_path = os.path.join(script_directory, 'assets/simulation.mp4')
fourcc = cv2.VideoWriter_fourcc(*'avc1') # Using avc1
FPS = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

# Initialize Deep SORT components
script_directory = os.getcwd()
model_path = os.path.join(script_directory, 'runs/detect/train/weights/best.pt')
model = YOLO(model_path)
model.iou = 0.45
max_cosine_distance = 0.3
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
detected_objects = pd.DataFrame(columns=['BoundingBox', 'ConfidenceScore', 'Time'])

# Training model and feature extractor
model_filename = os.path.join(os.path.dirname(__file__), '..', 'deep_sort', 'model_data', 'mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, input_name="images", output_name="features", batch_size=1)

# DEBUG Values
n_tracked = 0

#%%

start_time = time.time()

""" Main Simulation - Object Tracking """
try:
    # Loop through each frame in the video
    while cap.isOpened():

        # Read a frame from the video
        ret, frame_colored = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        '''
        if n_frames>=5:
            print('DEBUG')
        '''

        # Perform background subtraction (Remove basketball court)
        inpainted_frame = cv2.inpaint(frame_colored, mask, 1, cv2.INPAINT_TELEA)

        # Perform DeepSort (Object Tracking)
        inpainted_frame, n_tracked, detected_objects = object_tracking(inpainted_frame, model, tracker, encoder, n_tracked, detected_objects)

        # Display Video Frame
        cv2.imshow('Basketball Object Tracking', inpainted_frame)
        cv2.waitKey(1)  # Add a small delay to allow the window to update

        # Write the output frame
        out.write(inpainted_frame)

        # Increase frame count
        n_frames += 1 

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            logger.debug ("Simulation stopped through manual intervention")
            break
        
        # GitHub Actions specific code
        if os.getenv('GITHUB_ACTIONS') is True and n_frames > 0:
            logger.debug ("Simulation stopped, due to being tested in github actions")
            break

    # Log results summary
    n_objects = n_frames*11
    count_tracked_objects = n_tracked/n_objects*100

    logger.debug (f"Total number of objects that should have been tracked {n_objects}")
    logger.debug (f"Total number of objects tracked:, {count_tracked_objects:.4f}%")
    logger.debug ("Object Tracking succeeded")

except Exception as e:
    logger.error (f"An error occurred when processing the video frame: {e}")

finally:
    logger.debug ("Total time taken: %f seconds", time.time() - start_time)

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# %%
