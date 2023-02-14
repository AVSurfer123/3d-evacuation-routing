#detect doors and other relevant objects

# References:
#https://wellsr.com/python/object-detection-from-images-with-yolo/
#

import tensorflow_hub as hub
import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as pltp
from imageai.Detection import VideoObjectDetection

# Create a VideoCapture object and read from input file
vid1 = cv2.VideoCapture('video1.mp4')
 
# Check if camera opened successfully
if (vid1.isOpened()== False): 
  print("Error opening video stream or file")
 
# Create object of video object detection class
detector = VideoObjectDetection()

# Set model type for object detection, can also use any other model from imageai libbrary
detector.setModelTypeAsYOLOv3()

# Load model
detector.setModelPath("yolov3.pt")
detector.loadModel()

# Read until video is completed
while(vid1.isOpened()):
  # Capture frame-by-frame
  ret, frame = vid1.read()
  if ret == True:
    
    

    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
vid1.release()
 
# Closes all the frames
cv2.destroyAllWindows()