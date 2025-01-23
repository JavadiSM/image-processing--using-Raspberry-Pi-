import cv2
import numpy as np
from enums import enums
import time
def main(vid_cap:cv2.VideoCapture, fps = 1) -> None:
    cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('edges', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
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
        blank = np.zeros(frame.shape, dtype='uint8')
        result = see_frame(frame)
        countoures,hierarchies = cv2.findContours(result,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('computer eye',result)
        
        cv2.drawContours(blank,countoures,-1,(0,255,0),5)
        cv2.imshow('edges',blank)
        # Check for exit condition
        if exit_check(vid_cap):
            vid_cap.release()
            cv2.destroyAllWindows()
            break

def see_frame(frame):
    """
    computer only tends to see lines, borders, rather than photo
    """
    # gray = rescale(frame)
    gray = frame
    gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7), cv2.BORDER_DEFAULT)
    ret, threshold = cv2.threshold(blur,125,255,cv2.THRESH_BINARY)
    # DO NOT DELETE LINES COMMENTED I NEED THEM OK?
    # canny = cv2.Canny(blur,125,175)
    # dilated = cv2.dilate(canny,(15,20),iterations=10)
    # erdored = cv2.erode(dilated,(10,15), iterations=10)
    return threshold


def exit_check(vid_cap):
    if cv2.waitKey(10) & 0xFF == ord('q'):
        return True
    return False

def rescale(frame, scale: float = 1) -> np.ndarray:  
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
    # cv2.namedWindow('Live Feed')
    return cap

if __name__ == '__main__':
    print("Connecting to:", enums.URL.value)
    vid_cap = connect(enums.URL.value)
    main(vid_cap)
