import cv2
import numpy as np
from enums import enums
import time
def main(vid_cap:cv2.VideoCapture, fps = 1) -> None:
    """
    # READ BEFORE USE
    this function receives each image every time,
    if you are willing to change something, make a separate function and notice it should not stop
    nor make loop wait.
    """
    while True:
        # Read a frame from the stream
        _ , frame = vid_cap.read()
        """
        frame is to be proccessed.
        frame type is cv2.typing.MatLike
        """
        # proccess input to computers eye
        result = see_frame(frame)
        cv2.imshow('computer eye',result)
        
        # Check for exit condition
        if exit_check(vid_cap):
            vid_cap.release()
            cv2.destroyAllWindows()
            break

def see_frame(frame):
    """
    computer only tends to see lines, borders, rather than photo
    """
    frame = rescale(frame)
    blur = cv2.GaussianBlur(frame,(3,3), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blur,100,150)
    dilated = cv2.dilate(canny,(15,20),iterations=10)
    erdored = cv2.erode(dilated,(10,15), iterations=10)
    return erdored


def exit_check(vid_cap):
    if cv2.waitKey(10) & 0xFF == ord('x'):
        return True
    return False

def rescale(frame, scale: float = 0.5) -> np.ndarray:  
    # Get dimensions of the original frame  
    height, width = frame.shape[:2]  
    
    # Calculate new dimensions  
    new_dims = (int(width * scale), int(height * scale))  
    
    # Resize the frame  
    resized_frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)  
    
    return resized_frame 

def connect(ip_address) -> cv2.VideoCapture:
    """
    this function returns VideoCapture to input frames
    """
    cap = cv2.VideoCapture(ip_address)  # Start capturing from the IP camera

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()
    cv2.namedWindow('Live Feed')
    return cap

if __name__ == '__main__':
    print("Connecting to:", enums.URL.value)
    vid_cap = connect(enums.URL.value)
    main(vid_cap)
