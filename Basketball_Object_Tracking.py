'''
Basketball Object Tracking
This file tracks the positions/features of each player and the basketball.

Key Concepts Implemented:
- Hough Circle Transform - Object Detection specifically for Circles
- YOLO - End to End Object Object Detection using YOLO base for accuracy/speed balance
--> Implemented a custom model with 94.8% mAP50 (Refer CustomObjectDetection_Data/README.dataset.txt for more info)
- DeepSort - Multi Object Tracking Algorithm that handles well with occlusion
--> Implementing validation
'''

#%%

#Import Libraries

import cv2
import numpy as np
import os
import time
import logging
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import pandas as pd

# DeepSORT code from local files
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

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

def prepare_frame_for_display(frame):
    """
    Objective:

    
    Parameters:

    
    Returns:

    x
    """
        
    # If it's a PyTorch tensor
    if isinstance(frame, torch.Tensor):
        # Move to CPU and convert to numpy
        frame = frame.detach().cpu().numpy()
        
        # If it's a batch, take the first item
        if frame.ndim == 4:
            frame = frame[0]
        
        # Rearrange dimensions if necessary (CHW -> HWC)
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        
        # Scale to 0-255 if in float format
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
    
    # Ensure it's a numpy array
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a numpy array or PyTorch tensor")
    
    # Ensure it's in uint8 format
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # Ensure it's in HWC format
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        frame = frame.squeeze()
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    # Clip values to valid range
    frame = np.clip(frame, 0, 255)
    
    # Ensure the array is contiguous
    frame = np.ascontiguousarray(frame)
    
    return frame

#%%

def export_dataframe_to_csv(df, file_path):
    """
    Objective:
    Create a csv file of the objects tracked and its relevant features
    
    Parameters:
    [] df - Dataframe containing object's tracked and reelvant data
    [] file_path - File path of where the csv file should be saved at
    
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Export the DataFrame to CSV
        df.to_csv(file_path)
        print(f"DataFrame successfully exported to {file_path}")
    except Exception as e:
        print(f"An error occurred while exporting the DataFrame: {e}")


#%%

def object_tracking(frame, model, tracker, encoder, n_missed, detected_objects):
    """
    Objective:
    Perform deepsort object tracking on each video frame.
    
    Parameters:
    [array] frame - video frame before object tracking
    [class model] model - YOLO detection with the custom model
    [class deepsort] tracker - DeepSORT Tracker
    [function] encoder - Extracts relevant information (features) from the given frame 
    [int] n_missed - number of objects that are tracked (debugging purposes)
    [dataframe] detected_objects - pandas dataframe to store information on detected objects

    Returns:
    [array] frame - video frame after object tracking
    [int] n_missed - number of objects that are tracked (debugging purposes)
    [dataframe] detected_objects - pandas dataframe to store information on detected objects
    """
    
    """
    #Relevant parameters outside of object_tracking() which influences the accuracy of object tracking 
    tracker.py - max_iou_distance=0.5, max_age=2, n_init=3
    B_O_T.py - scores>0.65, max_cosine_distance=0.3, nn_metric=cosine
    --> model=YOLOv9c based custom model

    Check these if need more fine-tuning
    Kalman filter parameters
    max_dist
    """

    # Process the current frame with the YOLO model without gradient computation
    with torch.no_grad():
        results = model(frame)

    # Extract bounding boxes, scores, class_id (Basketball, Team_A, Team_B)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    # Convert class indices to class names
    class_names_dict = results[0].names
    class_names = [class_names_dict[int(i)] for i in class_ids]

    # Filter the detections based on confidence threshold
    mask = scores > 0.6 #Confidence Threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_names = [class_names[i] for i in range(len(class_names)) if mask[i]]
    
    # Compute features for Deep SORT
    features = encoder(frame, boxes)

    # Create detections for Deep SORT
    detections = []

    for box, score, feature, class_name in zip(boxes, scores, features, class_names):
        # Create a new Detection object
        detection = Detection(
            box,
            score,
            feature,
            class_name)
        detections.append(detection)
    
    if detections is None:
        print('No circle features were detected in the frame')
        logger.debug("No circle features were detected in the frame at:", np.float32(n_frames/30), "seconds")

    # Update tracker
    tracker.predict()
    tracker.update(detections)

    # Calculate number of objects tracked
    print('Number objects tracked:', len(tracker.tracks))
    n_missed += abs((len(tracker.tracks))-11)

    # Verify the tracks
    for ith_value, track in enumerate(tracker.tracks):
        try:
            # Calculate the object's detection confidence score
            confidence_score = scores[ith_value]
        
        except Exception as e:
            # Calculate the object's detection confidence score
            confidence_score = 0.0

        # Add a new detected object to the detected_objects dataframe
        ith_object_details = [
            track.track_id, #TrackID
            track.class_id, #ClassID - Basketball, Team_A, Team_B
            track.mean, #Track State - 8-dimensional vector: [x, y, a, h, vx, vy, va, vh]
            track.covariance, #Covariance between Track State variables
            confidence_score, #Confidence Score of object
            track.state, #Track Status - Tentative, Confirmed, Deleted
            track.hits, # Total objects successfully matched to the track
            track.age, # Total number of frames since the track was initialized
            track.features, # Features detected in the object
            np.float32(n_frames/30) #Time
        ]

        new_object = pd.DataFrame([ith_object_details], columns=
                                  ['TrackID', 'ClassID', 'Mean', 'Co-Variance', 'ConfidenceScore', 'State', 'Hits', 'Age', 'Features', 'Time'])
        detected_objects = pd.concat([detected_objects, new_object], ignore_index=True)

        # Check if
        # not track.is_confirmed()    : Check that an object track has not been found
        # track.time_since_update > 1 : Check the track has not been updated for more than one frame
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # Calculate the coordinates' border of the object
        bbox = track.to_tlbr()

        # Label color for detected object's track_id and confidence score
        color = (255, 255, 255)  # BGR format

        # Draw bounding boxes and IDs
        cv2.putText(frame, f"{track.track_id}-{confidence_score:.3f}", (int(bbox[0]), int(bbox[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, n_missed, detected_objects

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

if os.getenv('GITHUB_ACTIONS') is True:
    # If running through github actions update script_directory
    script_directory = os.getcwd()
    script_directory = script_directory.replace("/Basketball-PlayAnalysis/Basketball-PlayAnalysis", "/Basketball-PlayAnalysis", 1)
else:
    script_directory = os.getcwd()

video_path = os.path.join(script_directory, 'assets/simulation.mp4')
basketball_court_diagram = os.path.join(script_directory, 'assets/Basketball Court Diagram.jpg')

print(f"Video path: {video_path}")

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
param1_value = 12 # 12/13 - Best results
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
output_path = os.path.join(script_directory, 'assets/simulation_tracked.mp4')
fourcc = cv2.VideoWriter_fourcc(*'avc1') # Using avc1
FPS = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

# Transformation pipeline for each video frame for compatability
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),  # Resize to YOLO's expected input size
    transforms.ToTensor(),
])


# Switch model to GPU if available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU is being used")

# Initialize Deep SORT components
script_directory = os.getcwd()
model_path = os.path.join(script_directory, 'YOLOv10m_custom.pt')
model = YOLO(model_path)
model.to(device) # Move model to GPU
model.info() # Model Information
model.iou = 0.45
max_cosine_distance = 0.3
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

"""
# GPU Accelerated CuDF
detected_objects = cudf.DataFrame({
    'TrackID': cudf.Series([], dtype=np.int32),
    'ClassID': cudf.Series([], dtype=np.int32),
    'Mean': cudf.Series([], dtype='object'),
    'Co-Variance': cudf.Series([], dtype='object'),
    'ConfidenceScore': cudf.Series([], dtype=np.float32),
    'State': cudf.Series([], dtype=np.int32),
    'Hits': cudf.Series([], dtype=np.int32),
    'Age': cudf.Series([], dtype=np.int32),
    'Features': cudf.Series([], dtype='object'),
    'Time': cudf.Series([], dtype=np.float32)
})
"""

detected_objects = pd.DataFrame(columns=['TrackID', 'ClassID' , 'Mean', 'Co-Variance', 'ConfidenceScore', 'State', 'Hits', 'Age', 'Features', 'Time'])

# Training model and feature extractor
model_filename = os.path.join(os.path.dirname(__file__), '..', 'deep_sort', 'model_data', 'mars-small128.pb')
encoder = gdet.create_box_encoder(model_filename, input_name="images", output_name="features", batch_size=1)

# DEBUG Values
n_missed = 0
n_miscount = 0

#%%

start_time = time.time()

""" Main Simulation - Object Tracking """
try:
    # Loop through each frame in the video
    while cap.isOpened():

        # Read a frame from the video
        ret, frame_colored = cap.read()

        # If frame is read correctly ret is True
        if ret:
            #frame_colored = transform(frame_colored).unsqueeze(0).to(device)
            frame_colored = frame_colored
        if not ret:
            break

        '''
        if n_frames>=5:
            print('DEBUG')
        '''

        # Perform background subtraction (Remove basketball court)
        #inpainted_frame = cv2.inpaint(frame_colored, mask, 1, cv2.INPAINT_TELEA)

        # Preprocess Frame for optimal tracking
        #inpainted_frame = preprocess_frame(inpainted_frame, greyed = True, blur = 'median')

        # Perform DeepSort (Object Tracking)
        tracked_frame, n_missed, detected_objects = object_tracking(frame_colored, model, tracker, encoder, n_missed, detected_objects)


        # After processing your frame and before calling cv2.imshow

        #tracked_frame = prepare_frame_for_display(tracked_frame)

        # Display Video Frame
        cv2.imshow('Basketball Object Tracking', tracked_frame)
        cv2.waitKey(1)  # Add a small delay to allow the window to update

        # Write the output frame
        out.write(tracked_frame)

        # Increase frame count
        n_frames += 1 

        if n_frames > 120:
            break

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            logger.debug ("Simulation stopped through manual intervention")
            break
        
        # GitHub Actions specific code
        if os.getenv('GITHUB_ACTIONS') is True and n_frames > 0:
            logger.debug ("Simulation stopped, due to being tested in github actions")
            break

    # Export dataframe into csv
    file_path = os.path.join(os.getcwd(), 'assets', 'detected_objects.csv')
    export_dataframe_to_csv(detected_objects, file_path)

    # Log results summary
    n_objects = n_frames*11
    count_tracked_objects = (1 - (n_missed / (n_frames*11)))*100


    logger.debug (f"Total number of objects that should have been tracked {n_objects}")
    logger.debug (f"Percentage of objects tracked: {count_tracked_objects:.4f}%")
    logger.debug ("Object Tracking succeeded")
    print("Object Tracking succeeded")

except Exception as e:
    logger.error (f"An error occurred when processing the video frame: {e}")
    print("Object Tracking failed")

finally:
    logger.debug ("Total time taken: %f seconds", time.time() - start_time)

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# %%
