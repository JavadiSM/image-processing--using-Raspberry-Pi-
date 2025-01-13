import cv2
import numpy as np
from enum import Enum
import time

class enums(Enum):
    # Replace with your IP Webcam URL
    URL = 'http://192.168.101.154:8080/video'
    EXIT_KEY = ord('q')

def main(vid_cap, fps = 1):
    frame_delay = 1.0 / fps

    cv2.namedWindow('Live Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Live Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while True:
        # Read a frame from the stream
        ret, frame = vid_cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Apply image processing - Canny edge detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        # Display the resulting frame
        cv2.imshow('Live Feed', gray_frame)
        
        # Check for exit condition
        if exit_check(vid_cap):
            break


def exit_check(vid_cap):
    if cv2.waitKey(1) & 0xFF == enums.EXIT_KEY.value:
        vid_cap.release()
        cv2.destroyAllWindows()
        return True
    return False

def connect(ip_address):
    cap = cv2.VideoCapture(ip_address)  # Start capturing from the IP camera

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()
    return cap

if __name__ == '__main__':
    print("Connecting to:", enums.URL.value)
    vid_cap = connect(enums.URL.value)
    main(vid_cap)
