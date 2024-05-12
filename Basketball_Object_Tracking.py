# Offensive Basketball plays Object Tracking
# 
# Utilizing OpenCV, track players and basketball.

#%%

"Import Libraries"

import cv2
import numpy as np
import os
from ultralytics import YOLO

#%%

def circle_detection(p1, p2, results, frame_color):
    """
    Objective:
    Performs object detection using HoughCircles for each frame of the video
    Create a border and a center dot in the circle

    Parameters:
    [int]   p1 - param1 for HoughCircles()
    [int]   p2 - param2 for HoughCircles()
    [dict]  results - number of circles counted for the given param1, param2 values
    [array] frame_color - video frame

    Returns:
    [dict]  results - number of circles counted for the given param1, param2 values
    [array] frame_color - video frame

    """

    frame_greyed = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY) # Convert the image to grey scale for better OpenCV processing
    frame_greyed = cv2.medianBlur(frame_greyed, 5) # Apply median blur to reduce noise

    frame_hsv = cv2.cvtColor(frame_color, cv2.COLOR_BGR2HSV)
    
    circles_detected = cv2.HoughCircles(frame_greyed, cv2.HOUGH_GRADIENT, 1, minDist = 3, param1 = p1, param2 = p2, minRadius = 10, maxRadius = 20)

    # If no circles were detected, set circle_count to 0
    if circles_detected is None:
        circle_count = 0
    
    # If circles were detected, create border and a center dot around the detected circle 
    else:
        circles_detected = np.uint16(np.around(circles_detected))

        # Display for each detected circles
        for ith_circle in circles_detected[0, :]:

            x_coordinate = ith_circle[0]
            y_coordinate = ith_circle[1]
            radius = ith_circle[2]

            # Color of circle
            center_color_hue = frame_hsv[y_coordinate, x_coordinate][0] # Set color hue of the circle
            detection_color = color_detection(center_color_hue) # Set border color

            # Outer Circle
            cv2.circle(frame_color, (x_coordinate, y_coordinate), radius, detection_color, 2)

            # Center of circle
            cv2.circle(frame_color, (x_coordinate, y_coordinate), 2, detection_color, 3)
        
        circle_count = len(circles_detected[0])
    
    # Add the number of circles detected to the results
    results[(p1, p2)] += circle_count

    return results, frame_color


#%%

def color_detection(color_hue):
    """
    Objective:
    Returns the BGR value for a given hue color

    Parameters:
    [int] color_hue - Hue value

    Returns:
    [tuple] color_detected - Color detected is represented in (x, x, x) format
    """


    if color_hue < 10 or color_hue > 170:
        color_detected = (255, 0, 0)

    elif 10 <= color_hue <= 30:
        color_detected = (255, 165, 0)
    
    elif 100 <= color_hue <= 140:
        color_detected = (0, 0, 255)

    else:
        print(color_hue)
        color_detected = (0,0,0)

    return color_detected


#%%
# Path to the video file
script_directory = os.getcwd()
video_path = os.path.join(script_directory, 'simulation.mp4')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Parameter values to test
param1_values = [12]
param2_values = [15]

# Initialize results dictionary
resulting_values = {}

for param1 in param1_values:
    for param2 in param2_values:
        resulting_values[(param1, param2)] = 0

n_frames = 0 # Initialize n_frames to count the number of frames in the video

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

try: 
    # Loop through each frame in the video
    while cap.isOpened():
        # Read a frame from the video
        ret, frame_colored = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break
        
        # Test different param values in the for loop
        for param1 in param1_values:
            for param2 in param2_values:
                resulting_values, frame_colored  = circle_detection(param1, param2, resulting_values, frame_colored) # Perform circle detection

        # Display the frame
        cv2.imshow('Basketball Object Tracking', frame_colored)

        n_frames += 1 # Increase frame count

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    
    # Print summary of results
    print ("Total number of circles that should have been detected", n_frames*11)

    for (param1, param2), count in resulting_values.items():
        print(f"param1={param1}, param2={param2} -> {count} circles detected")


except:
    cap.release()
    cv2.destroyAllWindows()


# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


#%%
