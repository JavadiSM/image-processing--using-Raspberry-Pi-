import cv2  
import numpy as np  
from enums import enums

def main(vid_cap,url):
    while True:  
        # Read a frame from the stream  
        ret, frame = vid_cap.read()  
        if not ret:  
            print("Error: Could not read frame.")  
            break  
        # Display the resulting frames  
        cv2.imshow('Original Frame', frame)  # Show the original frame
        exit_check()
        

def exit_check():
    if cv2.waitKey(1) & 0xFF == enums.EXIT_KEY:
        vid_cap.release()  
        cv2.destroyAllWindows()

def connect(ip_address):
    url = ip_address
    # Start capturing from the IP camera  
    cap = cv2.VideoCapture(url)  

    if not cap.isOpened():  
        print("Yadet raft gooshio vasl koni dobare")  
        exit()
    return cap,url   

if __name__ == '__main__':    
    print(enums.URL.value)
    vid_cap,url = connect(enums.URL.value)
    main(vid_cap,url)