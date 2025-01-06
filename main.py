import cv2  
import numpy as np  

# Replace with your IP Webcam URL  
url = 'http://192.168.1.4:8080/video'  

# Start capturing from the IP camera  
cap = cv2.VideoCapture(url)  

if not cap.isOpened():  
    print("Error: Could not open video stream.")  
    exit()  

while True:  
    # Read a frame from the stream  
    ret, frame = cap.read()  
    if not ret:  
        print("Error: Could not read frame.")  
        break  

    # Perform image processing (example: convert to grayscale)  
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
    # Display the resulting frames  
    cv2.imshow('Original Frame', frame)  # Show the original frame  
    cv2.imshow('Gray Frame', gray_frame)  # Show the processed frame  

    # Exit on 'q' key  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

# Release the video capture object and close all OpenCV windows  
cap.release()  
cv2.destroyAllWindows()