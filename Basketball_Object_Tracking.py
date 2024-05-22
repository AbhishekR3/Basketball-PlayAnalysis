# Basketball Object Tracking

# Utilizing OpenCV, track players and basketball.

#%%

"Import Libraries"

import cv2
import numpy as np
import os
import time
import logging

#%%

def preprocess_frame(frame, greyed = True, blur = 'median'):
    """
    Objective:
    Preprocess frame of the video based on input parameters

    Parameters:
    [array] frame - video frame
    [bool] greyed (Default: True) - apply a grey filter on the image
    [string] blur (Default: 'median') - apply either median/gaussian blur

    Returns:
    [array] frame - video frame
    """

    # Convert the image to grey scale for better OpenCV processing
    if greyed is True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply blur to reduce noise
    if blur == 'median':
        frame = cv2.medianBlur(frame, 5)
    elif blur == 'gaussian':
        frame = cv2.GaussianBlur(frame, (5,5), 1.5)

    return frame


#%%

def circle_detection(p1, p2, results, frame_with_color):
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

        #"""
        # Display for each detected circles
        for ith_circle in circles_detected[0, :]:

            x_coordinate = ith_circle[0]
            y_coordinate = ith_circle[1]
            radius = ith_circle[2]

            # Color of circle
            center_color_hue = frame_hsv[y_coordinate, x_coordinate][0] # Set color hue of the circle
            detection_color = color_detection(center_color_hue) # Set border color

            # Outer Circle
            cv2.circle(frame_with_color, (x_coordinate, y_coordinate), radius, detection_color, 2)

            # Center of circle
            cv2.circle(frame_with_color, (x_coordinate, y_coordinate), 2, detection_color, 3)
        #"""

    # Add the number of circles detected to the results
    results[(p1, p2)] += circle_count

    return results, frame_with_color


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


#%% Configuring logging

log_file_path = 'object_tracking_output.log'

# Check if the file exists
if os.path.exists(log_file_path):
    # Delete the file
    os.remove(log_file_path)
    print(f"The file {log_file_path} has been deleted.")
else:
    print(f"No file found with the name {log_file_path}.")

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


#%% Initialize Simulation Variables

# Path to the video file / basketball court diagram
script_directory = os.getcwd()
video_path = os.path.join(script_directory, 'assets/simulation.mp4')
basketball_court_diagram = os.path.join(script_directory, 'assets/Basketball Court Diagram.jpg')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Create a mask to remove basketball court diagram from the video
mask = cv2.imread(basketball_court_diagram, cv2.IMREAD_GRAYSCALE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mask = cv2.resize(mask, (width, height))
_, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
mask = mask.astype(np.uint8)

# Parameter values to test
param1_values = [12] # 12/13 - Best results
param2_values = [15] # 15 - Best results

# Initialize results dictionary
resulting_values = {}

for param1 in param1_values:
    for param2 in param2_values:
        resulting_values[(param1, param2)] = 0

n_frames = 0 # Initialize n_frames to count the number of frames in the video

# Check if the video file opened successfully
if not cap.isOpened():
    logger.error ("Error: Couldn't open the video file.")
    exit()

# Create a VideoWriter object to save the output video
output_path = os.path.join(script_directory, 'assets/output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'avc1') # Using avc1
FPS = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

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

        # Inpaint the frame using the mask
        inpainted_frame = cv2.inpaint(frame_colored, mask, 1, cv2.INPAINT_TELEA)

        # Test different param values in the for loop
        for param1 in param1_values:
            for param2 in param2_values:
                resulting_values, inpainted_frame  = circle_detection(param1, param2, resulting_values, inpainted_frame) # Perform circle detection

        cv2.imshow('Basketball Object Tracking', inpainted_frame)

        out.write(inpainted_frame)

        n_frames += 1 # Increase frame count

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # For git actions testing, stop simulation to focus on testing code
        if n_frames > 0 and os.getenv('GITHUB_ACTIONS') is True:
            logger.debug("Simulation stopped, due to being tested in github actions")
            break

    # Log results summary
    logger.debug (f"Total number of circles that should have been detected {n_frames*11}%%")

    for (param1, param2), count in resulting_values.items():
        logger.debug (f"param1={param1}, param2={param2} -> {count} circles detected. {count/(n_frames*11)*100:.2f}%")

    logger.debug("Object Tracking succeeded")


except Exception as e:
    logger.error (f"An error occurred: {e}")

finally:
    logger.debug ("Total time taken: %f seconds", time.time() - start_time)
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()