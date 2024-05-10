#!/usr/bin/env python
# coding: utf-8

# Offensive Basketball plays Object Tracking
# 
# Utilizing OpenCV, track players and basketball.

# In[1]:


# Import Libraries

import cv2
import numpy as np
import os
from ultralytics import YOLO


# In[ ]:


# Import Variables from other files

get_ipython().run_line_magic('store', '-r PLAYER_RADIUS')
get_ipython().run_line_magic('store', '-r BALL_RADIUS')


# In[5]:


# Path to the video file
script_directory = os.getcwd()
video_path = os.path.join(script_directory, 'simulation.mp4')

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# Loop through each frame in the video
while cap.isOpened():
    # Read a frame from the video
    ret, frame_colored = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        break
        
    frame_greyed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the image to grey scale for better OpenCV processing
    frame_greyed = cv2.medianBlur(frame, 5) # Apply median blur to reduce noise

    
    circles_detected = cv2.HoughCircles(frame_greyed, cv2.HOUGH_GRADIENT, 1, 
                                        minDist = BALL_RADIUS/2, param1 = , param2 = , 
                                        minRadius = BALL_RADIUS - 2, maxRadius = PLAYER_RADIUS + 2)
    
    
    # Display the frame
    cv2.imshow('Frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


# In[3]:





# In[5]:


BALL_RADIUS


# In[ ]:




