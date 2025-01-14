from enum import Enum  
import cv2
import numpy as np  
class enums(Enum):
    # Replace with your IP Webcam URL  
    URL = 'http://192.168.1.6:8080/video'  
    EXIT_KEY = ord('q')