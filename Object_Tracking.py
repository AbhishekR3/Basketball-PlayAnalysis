'''
Basketball Object Tracking
This file tracks the positions/features of each player and the basketball.

Key Concepts Implemented:
- YOLO - End to End Object Object Detection using YOLO base for accuracy/speed balance
--> Implemented a custom model with 97.7% mAP50 (Refer CustomObjectDetection_Data/README.dataset.txt for more info)
- DeepSort - Multi Object Tracking Algorithm that handles well with occlusion
--> Validation Metrics can be found in Custom_Detection_Model/Object Tracking Metrics/MOT_Validation.py
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
    Prepare the frame for display, which is required due to GPU accelerated programming
    
    Parameters:
    [array]  frame - video frame
    
    Returns:
    [array]  frame - video frame    

    """
    try:
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

    except Exception as e:
        print(f"An error occurred when prepping frame for display: {e}")

#%%

def export_dataframe_to_csv(df, file_path):
    """
    Objective:
    Create a csv file of the objects tracked and its relevant features
    
    Parameters:
    [dataframe] df - Dataframe containing object's tracked and reelvant data
    [string] file_path - File path of where the csv file should be saved at
    
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

def export_validation_metrics(detected_objects):
    """
    Objective:
    Create a csv file of the objects tracked and its relevant features
    
    Parameters:
    [dataframe] df - Dataframe containing object's tracked and reelvant data
    [string] file_path - File path of where the csv file should be saved at
    
    """

    try:
        # Filter Metrics Objects
        validation_metrics_objects = detected_objects[['TrackID', 'Mean', 'ClassID', 'Frame']]

        validation_metrics_objects.loc[:, 'Mean'] = validation_metrics_objects['Mean'].apply(calculate_bbox) # Extract the boundingbox 

        validation_metrics_objects = validation_metrics_objects.rename(columns={'Mean': 'BBox'}) #Rename Mean to BBox for bounding box

        return validation_metrics_objects
    
    except Exception as e:
        print(f"An error occured while creating validation metrics: {e}")

#%%

def calculate_bbox(deepsort_mean):
    """
    Calculate bounding box coordinates from DeepSORT mean state.
    
    Parameters:
    deepsort_mean (np.array): Array of 8 values from DeepSORT.
    
    Returns:
    tuple: (top_left, top_right, bottom_left, bottom_right) coordinates.
    """
    try:
        # Split the string and store the values as floats
        deepsort_mean = deepsort_mean.strip('[]')
        deepsort_mean = [float(x) for x in deepsort_mean.split()]

        x, y, aspect_ratio, height = deepsort_mean[:4]

        # Calculate width from aspect ratio and height
        width = aspect_ratio * height

        # Calculate half width and half height
        half_width = width / 2
        half_height = height / 2

        # Calculate coordinates
        top_left = (x - half_width, y - half_height)
        bottom_right = (x + half_width, y + half_height)

        xtl = top_left[0]
        ytl = top_left[1]
        xbr = bottom_right[0]
        ybr = bottom_right[1]

        return np.array([xtl, ytl, xbr, ybr])

    except Exception as e:
        logger.error("An error occured when calculating bounding box: %s", e)

#%%

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
    
    '''
    #Relevant parameters outside of object_tracking() which influences the accuracy of object tracking 
    tracker.py - max_iou_distance=0.5, max_age=2, n_init=3
    B_O_T.py - scores>0.85, max_cosine_distance=0.3, nn_metric=cosine
    --> model=YOLOv10m based custom model

    Check these if need more fine-tuning
    Kalman filter parameters
    max_dist
    '''

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
        mask = filter_lowconfidence(class_names, scores, basketball_score=0.5, player_score=0.8) # Set confidence threshold for player and basketball, basketball is commonly occluded
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
            frame_time = np.float32(n_frames/30)
            logger.debug("No circle features were detected in the frame at:", frame_time)

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

            except:
                # Calculate the object's detection confidence score
                confidence_score = 0.0

            # Convert track.features to numpy array
            #track.features = track.features[0]

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
            color = (255, 255, 255)  # BGR format

            # Draw bounding boxes and IDs
            cv2.putText(frame, f"{track.track_id}-{confidence_score:.3f}", (int(bbox[0]), int(bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, n_missed, detected_objects
    
    except Exception as e:
        logger.error("Error in the object tracking: %s", e)

#%% Configure Docker containerization
try:
    base_dir = '/app'
    log_dir = os.environ.get('LOG_DIR', '/app/logs')
    tracking_dir = os.environ.get('TRACKING_DIR', '/app/tracking_data')
    assets_dir = os.environ.get('ASSETS_DIR', '/app/assets')
    video_dir = os.environ.get('VIDEO_DIR', '/app/simulations')
    deepsort_dir = os.environ.get('DeepSORT_DIR', '/app/deep_sort')

    # Set headless mode for OpenCV
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'

    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Use it before writing files
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

except Exception as e:
    print(f"Error in creating environment for containers: {e}")
    raise
#%% Configuring logging

try:
    log_file_path = os.path.join(log_dir, 'tracking.log')

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
video_path = os.path.join(video_dir, 'simulation_video.mp4')
basketball_court_diagram = os.path.join(assets_dir, 'Basketball Court Diagram.jpg')

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
output_path = os.path.join(video_dir, 'simulation_tracked.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Using avc1
FPS = cap.get(cv2.CAP_PROP_FPS)
try:
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
except:
    out = cv2.VideoWriter(output_path, fourcc, FPS, (470, 500))

# Transformation pipeline for each video frame for compatability
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 640)),  # Resize to YOLO's expected input size
    transforms.ToTensor(),
])

'''
# Set model to GPU/CPU depending on environemnt
# If testing through github actions set to CPU
if os.getenv('GITHUB_ACTIONS') == 'true':
    device = torch.device("cpu")
    print("CPU is being used")
# If GPU is available:
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("GPU is being used")
'''

# Initialize Deep SORT components
#script_directory = os.getcwd()
model_path = os.path.join(assets_dir, 'YOLOv10m_custom.pt')
#model_path = os.path.join(script_directory, 'runs/detect/train/weights/best.pt')
model = YOLO(model_path)
#model.to(device) # Move model to GPU
model.info() # Model Information
model.iou = 0.45
max_cosine_distance = 0.4
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

detected_objects = pd.DataFrame(columns=['TrackID', 'ClassID' , 'Mean', 'Co-Variance', 'ConfidenceScore', 'State', 'Hits', 'Age', 'Features', 'Frame'])

# Training model and feature extractor
model_filename = os.path.join(deepsort_dir, 'model_data/mars-small128.pb')
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
        

        # Perform background subtraction (Remove basketball court)
        #inpainted_frame = cv2.inpaint(frame_colored, mask, 1, cv2.INPAINT_TELEA)

        # Preprocess Frame for optimal tracking
        inpainted_frame = preprocess_frame(frame_colored, greyed = True, blur = 'median')
        '''
        
        # Perform DeepSort (Object Tracking)
        tracked_frame, n_missed, detected_objects = object_tracking(frame_colored, model, tracker, encoder, n_missed, detected_objects)


        # After processing your frame and before calling cv2.imshow

        #tracked_frame = prepare_frame_for_display(tracked_frame)

        # Display Video Frame
        #cv2.imshow('Basketball Object Tracking', tracked_frame)
        cv2.waitKey(1)  # Add a small delay to allow the window to update

        # Write the output frame
        #out.write(tracked_frame)

        # Increase frame count
        n_frames += 1
        print('Frame number:', n_frames)

        # If 3 frames has been processed and present in Docker Environment, break
        if n_frames > 3 and os.path.exists('/.dockerenv'):
            break

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            logger.debug ("Simulation stopped through manual intervention")
            break
        
        # GitHub Actions specific code
        if os.getenv('GITHUB_ACTIONS') == 'true' and n_frames > 0:
            logger.debug ("Simulation stopped, due to being tested in github actions")
            break

    # Export extracted features to dataframe into csv
    detectedobjects_file_path = os.path.join(tracking_dir, 'detected_objects.csv')
    export_dataframe_to_csv(detected_objects, detectedobjects_file_path)

    '''
    # Export MOT validation metrics dataframe into csv
    detected_objects = pd.read_csv(os.path.join(os.getcwd(), 'assets', 'detected_objects.csv'))
    MOTvalidation_file_path = os.path.join(os.getcwd(), 'Custom_Detection_Model', 'Object Tracking Metrics', 'MOT_validationmetrics.csv')
    detected_objects_filtered = export_validation_metrics(detected_objects)
    export_dataframe_to_csv(detected_objects_filtered, MOTvalidation_file_path)
    '''

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