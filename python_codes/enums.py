from enum import Enum  
import cv2  
import numpy as np  
class enums(Enum):
    # Replace with your IP Webcam URL  
    URL = 'http://192.168.101.154:8080/video'  
    EXIT_KEY = ord('q')