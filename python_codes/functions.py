import cv2  
import numpy as np

def rescale(frame, scale: float = 0.75) -> np.ndarray:  
    height, width = frame.shape[:2]  
    new_dims = (int(width * scale), int(height * scale))  
    resized_frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)  
    return resized_frame

def process_scale_frame(frame, canny_low = 0, canny_high = 30, kernel_size = 9):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red in HSV
    lower_red1 = np.array([0, 120, 70])  # Red hue part 1
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])  # Red hue part 2
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_areas = cv2.bitwise_and(frame, frame, mask=red_mask)

    gray = cv2.cvtColor(red_areas, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, canny_low, canny_high)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_cleaned = cv2.dilate(edges, kernel, iterations=1)
    edges_cleaned = cv2.erode(edges_cleaned, kernel, iterations=1)

    return edges_cleaned

def calculate_scale(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_area = 0
    scale_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > scale_area:
            scale_area = area
            scale_contour = cnt

    return scale_area, scale_contour

def process_area_frame(frame, canny_low=0, canny_high=40, kernel_size=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    edges = cv2.Canny(filtered, canny_low, canny_high)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_cleaned = cv2.dilate(edges, kernel, iterations=1)
    edges_cleaned = cv2.erode(edges_cleaned, kernel, iterations=1)

    return edges_cleaned

def calculate_area(frame, scale):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    valid_contours = []

    for contour in contours:
        epsilon = 0.035 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        area = cv2.contourArea(approx)

        if area > scale * 0.5:
            areas.append(area)
            valid_contours.append(approx)

    scaled_areas = [area / scale for area in areas]

    return scaled_areas, valid_contours
