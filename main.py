import cv2  
import numpy as np  
import enums

def main(vid_cap,url):
    while True:  
        # Read a frame from the stream  
        ret, frame = vid_cap.read()  
        if not ret:  
            print("Error: Could not read frame.")  
            break  

        # Perform image processing (example: convert to grayscale)  
        show_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
        # Display the resulting frames  
        cv2.imshow('Original Frame', frame)  # Show the original frame  
        cv2.imshow('Gray Frame', show_frame)  # Show the processed frame  

        # Exit on 'q' key  
        if cv2.waitKey(1) & 0xFF == enums.EXIT_KEY:  
            break 

def connect(ip_address):
    url = ip_address
    # Start capturing from the IP camera  
    cap = cv2.VideoCapture(url)  

    if not cap.isOpened():  
        print("Yadet raft gooshio vasl koni dobare")  
        exit()
    return cap,url   

if __name__ == '__main__':    
    vid_cap,url = connect(enums.URL)
    # Release the video capture object and close all OpenCV windows  
    main(vid_cap,url)
    vid_cap.release()  
    cv2.destroyAllWindows()