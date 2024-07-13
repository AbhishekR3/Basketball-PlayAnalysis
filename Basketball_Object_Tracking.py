'''
Basketball Object Tracking
This file tracks the positions of each player and the basketball.

Main ML related algorithms used are:
- Hough Circle Transform - Object Detection specifically for Circles
- YOLOv9 - End to End Object Object Detection
- DeepSort - Multi Object Tracking Algorithm that factors in occlusion
'''

'''
Upcoming Implementations:

1.
# Features extracted through object detection (Hough Circle Transformation)
tracker.py
self.tracks = []

Include speed and direction as a feature

2.
Game_Simulation.py
Update place_circle_with_constraints
Create a perimeter and center circle related to the circle

3.
Game_Simulation.py
Update game simulation to include alpha blending

4.
YOLOv10 implementation

5.
Optimize parameters

Grid Search
- Kalman filter parameters
- max_dist
- NMS threshold? - Removed in YOLOv10
- other parameters

6.
Feature Extractor Model
- ResNet50
- EfficientNet

'''

#%%

"Import Libraries"

import cv2
import numpy as np
import os
import time
import logging
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO


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

#%%

def color_detection(color_hue):
    """
    Objective:
    Returns the BGR value for a given hue color.
    Switch Red and Blue values to create a distnct circle around the players.

    Parameters:
    [int] color_hue - Hue value

    Returns:
    [tuple] color_detected - Color detected is represented in (x, x, x) format
    """

    try:
        # Red detected -> Return BLue
        if color_hue < 10 or color_hue > 170:
            color_detected = (255, 0, 0)

        # Orange
        elif 10 <= color_hue <= 30:
            color_detected = (255, 165, 0)

        # Blue
        elif 100 <= color_hue <= 140:
            color_detected = (0, 0, 255)

        else:
            logger.debug("Color hue outside of range: %s", color_hue)
            color_detected = (0,0,0)

        return color_detected

    except Exception as e:
        logger.error("Error in detecting color: %s", e)


#%%

def create_optimal_tracking_color(object_bgr):
    """
    Objective: 
    Return the perimeter color and center color of the detected circle. 
    This is to improve the detection object tracking.
    Perimeter color is a darker shade. Center color is a lighter shade.

    Parameters:
    [tuple] object_bgr - object color in bgr

    
    Returns:
    [tuple] perimeter_color - perimeter color in rgb
    [tuple] center_color - center color in rgb
    """

    try:
        #Ensuring the color stays within the 0-255 color range
        def adjust_color_value(color_value, factor):
            return min(max(int(color_value * factor), 0), 255)

        b, g, r = object_bgr

        # Calculate the darker shade for the perimeter
        perimeter_color = (
            adjust_color_value(r, 0.7),
            adjust_color_value(g, 0.7),
            adjust_color_value(b, 0.7)
        )

        # Calculate the lighter shade for the center
        center_color = (
            adjust_color_value(r + (255 - r) * 0.5, 1),
            adjust_color_value(g + (255 - g) * 0.5, 1),
            adjust_color_value(b + (255 - b) * 0.5, 1)
        )

    except Exception as e:
        logger.error("Error in calculating optimal color for object detection: %s", e)

    return perimeter_color, center_color


#%%

def object_detection(p1, p2, results, frame_with_color):
    """
    Objective:
    Performs object detection using HoughCircles for each frame of the video
    Create a border and a center dot in the circle

    Parameters:
    [int]   p1 - param1 for HoughCircles()
    [int]   p2 - param2 for HoughCircles()
    [dict]  results - number of circles counted for the given param1, param2 values
    [array] frame_with_color - video frame

    Returns:
    [dict]  results - number of circles counted for the given param1, param2 values
    [array] frame_with_color - video frame
    """

    try:
        frame_greyed = preprocess_frame(frame_with_color)

        frame_hsv = cv2.cvtColor(frame_with_color, cv2.COLOR_BGR2HSV)

        circles_detected = cv2.HoughCircles(frame_greyed, cv2.HOUGH_GRADIENT, 1, minDist = 3, param1 = p1, param2 = p2, minRadius = 10, maxRadius = 20)

        # If no circles were detected, set circle_count to 0
        if circles_detected is None:
            circle_count = 0

        # If circles were detected, create border and a center dot around the detected circle
        else:
            circles_detected = np.uint16(np.around(circles_detected))
            circle_count = len(circles_detected[0])

            detections = []

            # Display for each detected circles
            for ith_circle in circles_detected[0, :]:

                # Detect coordinates and radius of circles
                x_coordinate = ith_circle[0]
                y_coordinate = ith_circle[1]
                radius = ith_circle[2]

                # Color of circle
                center_color_hue = frame_hsv[y_coordinate, x_coordinate][0] # Set color hue of the circle
                detection_color = color_detection(center_color_hue) # Set border color

                #Set the optimal color for tracking purposes
                perimeter_detection_color, center_detection_color = create_optimal_tracking_color(detection_color)

                # Outer Circle
                cv2.circle(frame_with_color, (x_coordinate, y_coordinate), radius, perimeter_detection_color, 2)

                # Center of circle
                cv2.circle(frame_with_color, (x_coordinate, y_coordinate), 2, center_detection_color, 3)

                print(x_coordinate, 
                      y_coordinate,
                      radius,
                      center_color_hue,
                      detection_color)

                '''
                # Set detections from deep_sort algorithm
                detections.append(Detection([x_coordinate - radius, y_coordinate - radius, x_coordinate + radius, y_coordinate + radius], confidence=0.8, feature=None))


            # Update tracker
            tracker.predict()
            tracker.update(detections)


            # Draw tracking results
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame_with_color, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame_with_color, f'ID: {track.track_id}', (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            '''

        # Add the number of circles detected to the results
        results[(p1, p2)] += circle_count

        return results, frame_with_color

    except Exception as e:
        logger.error("Error in circle detection: %s", e)


#%%

def object_tracking(frame, model, tracker, encoder, n_tracked):
    """
    Objective:
    Perform deepsort object tracking on each video frame.
    
    Parameters:
    [array] frame - video frame before object tracking
    [class model] model - YOLO detection with the pre-trained model
    [class deepsort] tracker - DeepSORT Tracker
    [function] encoder - Extracts relevant information (features) from the given frame 
    [int] n_tracked - number of objects that are tracked (debugging purposes)

    Returns:
    [array] frame - video frame after object tracking
    [int] n_tracked - number of objects that are tracked (debugging purposes)
    """

    results = model(frame)

    # Extract bounding boxes and scores
    boxes = results[0].boxes.xyxy.numpy()
    scores = results[0].boxes.conf.numpy()
    
    # Filter the detections based on confidence threshold
    mask = scores > 0.2
    boxes = boxes[mask]
    scores = scores[mask]

    # Calculate number of objects detected
    print('Detected Object')
    print(boxes)

    if boxes is not None:
        n_tracked += len(boxes)

    # Relevant parameters
    #tracker.py - max_iou_distance=0.2, max_age=5, n_init=5
    #B_O_T.py - scores>0.2, max_cosine_distance=0.5, nn_metric=euclidean/cosine (need to test further), model=YOLOv9
    #63.0567%

    #Kalman filter parameters
    #max_dist
    #NMS threshold


    # Compute features for Deep SORT
    features = encoder(frame, boxes)

    # Create detections for Deep SORT
    detections = [Detection(box, score, feature) for box, score, feature in zip(boxes, scores, features)]

    # Update tracker
    tracker.predict()
    tracker.update(detections)

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

            # Set object border lines + object's track_id and confidence score
            color = (255, 255, 255)  # BGR format
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"{track.track_id}-{confidence_score:3f}", (int(bbox[0]), int(bbox[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        except:
            # Calculate the object's detection confidence score
            confidence_score = 0.0
            #BUG: Why is confidence_score not recognized but object is still detected?
            
            # Set object border lines + object's track_id and confidence score
            color = (255, 255, 255)  # BGR format
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"{track.track_id}-{confidence_score:3f}", (int(bbox[0]), int(bbox[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, n_tracked

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
video_path = os.path.join(script_directory, 'assets/simulation.mp4')
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
output_path = os.path.join(script_directory, 'assets/object_tracking_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'avc1') # Using avc1
FPS = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))


# Initialize Deep SORT components
model = YOLO("yolov9c.pt")
max_cosine_distance = 0.5
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Adjust this path based on the location of your script relative to the model file
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

        # Perform background subtraction (Remove basketball court)
        inpainted_frame = cv2.inpaint(frame_colored, mask, 1, cv2.INPAINT_TELEA)

        # Perform Hough Circles detection (Object Detection)
        resulting_values, inpainted_frame  = object_detection(param1_value, param2_value, resulting_values, inpainted_frame) 

        # Perform DeepSort (Object Tracking)
        inpainted_frame, n_tracked = object_tracking(inpainted_frame, model, tracker, encoder, n_tracked)

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

        '''
        For git actions testing, stop simulation to focus on testing code. 
        This should be commented out

        if n_frames > 0 and os.getenv('GITHUB_ACTIONS') is True:
            logger.debug ("Simulation stopped, due to being tested in github actions")
            break
        '''
    # Log results summary
    n_objects = n_frames*11
    count_detected_objects = list(resulting_values.values())[0]/n_objects*100
    count_tracked_objects = n_tracked/n_objects*100

    logger.debug (f"Total number of objects that should have been detected {n_objects}")
    logger.debug (f"Total number of objects that was detected: {count_detected_objects:.2f}%")
    logger.debug (f"Total number of objects tracked:, {count_tracked_objects:.4f}%")

    logger.debug ("Object Tracking succeeded")

except Exception as e:
    logger.error (f"An error occurred when processing the video frame: {e}")

finally:
    logger.debug ("Total time taken: %f seconds", time.time() - start_time)

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()