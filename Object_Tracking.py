'''
Basketball Object Tracking
This file tracks the positions/features of each player and the basketball.

Key Concepts Implemented:
- YOLO - End to End Object Object Detection using YOLO base for accuracy/speed balance
--> Implemented a custom model with 99.96% mAP50 (Refer References/Custom_DetectionModel.txt for more info)
- DeepSort - Multi Object Tracking Algorithm that handles well with occlusions
'''

#%% Import Statements

#Import Libraries

import cv2
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import pandas as pd
from utils import export_dataframe_to_csv, configure_logger

# DeepSORT code from local files
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

#%% Filter Low Confidence Detections

def filter_lowconfidence(class_names, scores, basketball_score=0.5, player_score=0.8):
    '''
    Objective:


    Parameters:
    [array] scores
    [float]
    [float]

    Returns:
    [array] mask - Array of boolean values on which values to remove 
    '''

    try:
        mask = []
        result = np.column_stack((class_names, scores)) #Combine into 2D array

        for ith in result:
            if ith[0] == 'Basketball':
                if float(ith[1]) > basketball_score:
                    mask.append(True)
                else:
                    mask.append(False)
            else:
                if float(ith[1]) > player_score:
                    mask.append(True)
                else:
                    mask.append(False)
        return mask

    except Exception as e:
        logger.error("Error in filtering: %s", e)

#%% Object Tracking with DeepSORT

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

    try:
        # Process the current frame with the YOLO model without gradient computation
        with torch.no_grad():
            results = model(frame)
        
        print(results)

        # Extract bounding boxes, scores, class_id (Basketball, Team_A, Team_B)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        # Convert class indices to class names
        class_names_dict = results[0].names
        class_names = np.array([class_names_dict[int(i)] for i in class_ids])

        # Filter the detections based on confidence threshold        
        mask = filter_lowconfidence(class_names, scores, basketball_score=0.5, player_score=0.6) # Set confidence threshold for player and basketball, basketball is commonly occluded
        boxes = boxes[mask]
        scores = scores[mask]
        class_names = [class_names[i] for i in range(len(class_names)) if mask[i]]
        
        # Compute features for DeepSORT
        features = encoder(frame, boxes)

        # Create detections for DeepSORT
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
            frame_time = np.float32(n_frames/30)
            logger.debug(f"No circle features were detected in the frame at: {frame_time}")

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
                logger.debug("Error in calculating confidence score: %s", e)
                confidence_score = 0.0


            # Add a new detected object to the detected_objects dataframe
            ith_object_details = [
                int(track.track_id)-1, #TrackID
                track.class_id, #ClassID - Basketball, Team_A, Team_B
                track.mean, #Track State - 8-dimensional vector: [x, y, a, h, vx, vy, va, vh]
                track.covariance, #Covariance between Track State variables
                confidence_score, #Confidence Score of object
                track.state, #Track Status - Tentative, Confirmed, Deleted
                track.hits, # Total objects successfully matched to the track
                track.age, # Total number of frames since the track was initialized
                track.features, # Features detected in the object
                n_frames #Nth Frame
            ]

            # Create a new DataFrame for the detected object
            new_object = pd.DataFrame([ith_object_details], columns=
                                    ['TrackID', 'ClassID', 'Mean', 'Co-Variance', 'ConfidenceScore', 'State', 'Hits', 'Age', 'Features', 'Frame'])
            
            detected_objects = pd.concat([detected_objects, new_object], ignore_index=True)

            # Check if
            # not track.is_confirmed()    : Check that an object track has not been found
            # track.time_since_update > 1 : Check the track has not been updated for more than one frame
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Calculate the coordinates' border of the object
            bbox = track.to_tlbr()

            # Label color for detected object's track_id and confidence score
            color = (255, 255, 255)  # White in BGR format
            
            # Draw bounding boxes and IDs
            cv2.putText(frame, f"{track.track_id}-{confidence_score:.3f}", (int(bbox[0]), int(bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        print('frame completed')

        return frame, n_missed, detected_objects
    
    except Exception as e:
        logger.error("Error in the object tracking: %s", e)

#%% Configure Docker containerization
#'''
try:
    # Set directories for Docker environment
    base_dir = '/app'
    log_dir = os.environ.get('LOG_DIR', '/app/logs')
    tracking_dir = os.environ.get('TRACKING_DIR', '/app/tracking_data')
    assets_dir = os.environ.get('ASSETS_DIR', '/app/assets')
    video_dir = os.environ.get('VIDEO_DIR', '/app/simulations')
    deepsort_dir = os.environ.get('DeepSORT_DIR', '/app/deep_sort')

    # Ensure directories exist
    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    ensure_dir(log_dir)
    ensure_dir(tracking_dir)
    ensure_dir(assets_dir)
    ensure_dir(video_dir)
    ensure_dir(deepsort_dir)

    print('Base Directory:', base_dir)
    print('Log Directory:', log_dir)
    print('Tracking Directory:', tracking_dir)
    print('Assets Directory:', assets_dir)
    print('Simulations Directory:', video_dir)
    print('DeepSORT Directory:', deepsort_dir)

    # Set headless mode for OpenCV
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'


except Exception as e:
    print(f"Error in creating environment for containers: {e}")
    raise
#'''

#%% Initialize Simulation Variables

# Configuring logging
logger = configure_logger('tracking')

# Path to the video file / basketball court diagram
try:
    video_path = os.path.join(video_dir, 'simulation_video.mp4') # Passing simulation video path
    #video_path = os.path.join(video_dir, 'random_movement_video.mp4') # Random movement video path
    basketball_court_diagram = os.path.join(assets_dir, 'Basketball Court Diagram.jpg')
except Exception as e:
    video_path = "/Users/abhishekramesh/Desktop/simulation_video.mp4" # Passing simulation video path
    script_directory = os.getcwd()
    #video_path = os.path.join(script_directory, 'simulations', 'random_movement_video.mp4') # Random movement video path
    basketball_court_diagram = "/Users/abhishekramesh/Library/Mobile Documents/com~apple~CloudDocs/Basketball-PlayAnalysis/assets/Basketball Court Diagram.jpg"

print(f"Video path: {video_path}")

# Open the video file
cap = cv2.VideoCapture(video_path)

'''
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
'''

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
try:
    output_path = os.path.join(video_dir, 'simulation_tracked.mp4')
except Exception as e:
    output_path = "/Users/abhishekramesh/Desktop/simulation_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Using avc1
FPS = cap.get(cv2.CAP_PROP_FPS)
try:
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
except Exception as e:
    print(f"Error in establishing width, height in video writer: {e}")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (470, 500))

# Transformation pipeline for each video frame for compatability
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),  # Resize to YOLO's expected input size
    transforms.ToTensor(),
])

#%% Initialize YOLO and DeepSORT

try:
    model_path = os.path.join(assets_dir, 'YOLOv10s_custom.pt')
    print("Model path created")
except Exception as e:
    model_path = "/Users/abhishekramesh/Library/Mobile Documents/com~apple~CloudDocs/Basketball-PlayAnalysis/assets/YOLOv10s_custom.pt"
model = YOLO(model_path)
model.info() # Model Information
model.iou = 0.45
max_cosine_distance = 0.4
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

detected_objects = pd.DataFrame(columns=['TrackID', 'ClassID' , 'Mean', 'Co-Variance', 'ConfidenceScore', 'State', 'Hits', 'Age', 'Features', 'Frame'])

# Training model and feature extractor for DeepSORT
try:
    model_filename = os.path.join(deepsort_dir, 'model_data/mars-small128.pb')
except Exception as e:
    model_filename = "/Users/abhishekramesh/Library/Mobile Documents/com~apple~CloudDocs/Basketball-PlayAnalysis/deep_sort/model_data/mars-small128.pb"

encoder = gdet.create_box_encoder(
    model_filename, 
    input_name="images", 
    output_name="features", 
    batch_size=1
)

# DEBUG Values
n_missed = 0
n_miscount = 0

#%% Perform Object Tracking

start_time = time.time()

""" Main Simulation - Object Tracking """
try:

    # Loop through each frame in the video
    while cap.isOpened():
        print('Start processing frame')

        # Read a frame from the video
        ret, frame_colored = cap.read()

        # If frame is read correctly ret is True
        if ret:
            frame_colored = frame_colored

        if not ret:
            break
        
        # Perform DeepSort (Object Tracking)
        tracked_frame, n_missed, detected_objects = object_tracking(frame_colored, model, tracker, encoder, n_missed, detected_objects)
        print('Object Tracking completed')

        # Prepare the frame for display
        #tracked_frame = prepare_frame_for_display(tracked_frame)

        # Display Video Frame
        #cv2.imshow('Basketball Object Tracking', tracked_frame)
        #cv2.waitKey(1)  # Add a small delay to allow the window to update

        # Write the output frame
        #out.write(tracked_frame)

        # Increase frame count
        n_frames += 1
        print('Frame number:', n_frames)

        '''
        # If 5 frames has been processed and present in Docker Environment, break
        if n_frames > 5 and os.path.exists('/.dockerenv'):
            print('Simulation stopped, due to being tested in docker environment')
            break
        
        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            logger.debug ("Simulation stopped through manual intervention")
            break
        
        # GitHub Actions specific code
        if os.getenv('GITHUB_ACTIONS') == 'true' and n_frames > 0:
            logger.debug ("Simulation stopped, due to being tested in github actions")
            break
        '''

    # If no objects were detected in the video, log an error and exit
    if detected_objects.empty:
        logger.error ("No objects were detected in the video")
        print("No objects were detected in the video")
        exit()

    # Export extracted features to dataframe into csv
    detectedobjects_file_path = os.path.join(tracking_dir, 'detected_objects.csv')
    export_dataframe_to_csv(detected_objects, detectedobjects_file_path, logger)
    print('Exported detected objects to csv')

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
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        pass