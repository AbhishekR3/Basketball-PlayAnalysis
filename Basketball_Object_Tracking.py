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
    """

    frame_greyed = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY) # Convert the image to grey scale for better OpenCV processing
    frame_greyed = cv2.medianBlur(frame_greyed, 5) # Apply median blur to reduce noise
    
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

            # Outer Circle
            cv2.circle(frame_color, (x_coordinate, y_coordinate), radius, (0,255,0), 2)

            # Center of circle
            cv2.circle(frame_color, (x_coordinate, y_coordinate), 3, (0,255,0), 3)
        
        circle_count = len(circles_detected[0])
    
    # Add the number of circles detected to the results
    results[(p1, p2)] += circle_count

    return results, frame_color


#%%
# Path to the video file
script_directory = os.getcwd()
video_path = os.path.join(script_directory, 'simulation.mp4')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Parameter values to test
param1_values = [12.5]
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
        
        for param1 in param1_values:
            for param2 in param2_values:
                resulting_values, frame_colored  = circle_detection(param1, param2, resulting_values, frame_colored) # Perform circle detection

        # Display the frame
        cv2.imshow('Basketball Object Tracking', frame_colored)

        n_frames += 1

        # Press 'q' to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    
    # Print summary of results
    print ("Total Circles", n_frames*11)

    for (param1, param2), count in resulting_values.items():
        print(f"param1={param1}, param2={param2} -> {count} circles detected")


except:
    cap.release()
    cv2.destroyAllWindows()


# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


#%%
