import cv2
import numpy as np

# Global variable to hold the last detected red object as a template
last_template = None
def calculate_area(frame, scale):  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    areas = []  
    cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Threshold for minimum area  
            areas.append(area / (scale ** 2))  # Calculate the area relative to the scale
            cnts.append(cnt)
    return areas


def rescale(frame, scale: float = 0.8) -> np.ndarray:  
    # Get dimensions of the original frame  
    height, width = frame.shape[:2]  
    
    # Calculate new dimensions  
    new_dims = (int(width * scale), int(height * scale))  
    
    # Resize the frame
    resized_frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)  
    
    return resized_frame


def detect_red_and_blackout(image):  
    """Detect red areas in the image and blackout everything else."""  
    
    # Convert the image from BGR to HSV color space  
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  

    # Define tighter HSV range for red color  
    # Lower red range: hue between 0 and 10  
    lower_red1 = np.array([0, 120, 70])   # Adjusted lower range (increased saturation/value)  
    upper_red1 = np.array([10, 255, 255]) # Upper range   
    # Higher red range: hue between 170 and 180  
    lower_red2 = np.array([170, 120, 70]) # Adjusted lower range (increased saturation/value)  
    upper_red2 = np.array([180, 255, 255]) # Upper range   

    # Create masks for detecting red in the defined ranges  
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)  
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)  

    # Combine the masks  
    red_mask = cv2.bitwise_or(mask1, mask2)  

    # Create an output image to keep only red areas and make everything else black  
    output_image = np.zeros_like(image)  
    output_image[red_mask > 0] = image[red_mask > 0]  

    return output_image


def find_contours(frame):  
    """Detect contours of objects in the given frame."""  
    # Convert the frame to grayscale  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
    # Apply Gaussian blur to reduce noise and improve contour detection  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
    
    # Apply edge detection  
    edged = cv2.Canny(blurred, 50, 150)  
    
    # Find contours in the edged image  
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    
    return contours  

def get_most_similar_contour(reference_contour, contours):  
    """Finds the most similar contour to the reference contour based on shape matching."""  
    best_match = None  
    best_match_value = float('inf')  
    
    for contour in contours:  
        if len(contour) >= 5:  # Shape matching requires at least 5 points  
            # Calculate Hu Moments to compare the shapes  
            reference_moments = cv2.HuMoments(cv2.moments(reference_contour)).flatten()  
            contour_moments = cv2.HuMoments(cv2.moments(contour)).flatten()  
            # Calculate the distance using log transformation to compare the moments  
            distance = np.sqrt(np.sum((reference_moments - contour_moments) ** 2))  
            
            # Update best match if the current contour is more similar (lower distance)  
            if distance < best_match_value:  
                best_match_value = distance  
                best_match = contour  
    
    return best_match  


def similarity_detector(reference_frame):  
    """Create a function to detect most similar objects in subsequent frames."""  
    # Get contours from reference frame  
    reference_contours = find_contours(reference_frame)  
    
    if not reference_contours:  # If no objects are detected  
        return None  
    
    # For simplicity, we will take the first contour as the reference  
    reference_contour = max(reference_contours, key=cv2.contourArea)  

    def detect_in_frame(frame):  
        """Detects most similar object based on the reference contour in a new frame."""  
        new_contours = find_contours(frame)  
        matching_contour = get_most_similar_contour(reference_contour, new_contours)  
        return matching_contour  # Returns the contour of the most similar object  
    
    return detect_in_frame

def calculate_length_and_area(contour):
    """Calculate the length of the contour and area of the shape."""
    length = cv2.arcLength(contour, True)  # Length of the contour
    area = cv2.contourArea(contour)       # Area of the contour
    return length, area

def feature_based_matching(frame, template):
    """Use ORB feature matching to find the best match for the template in the frame."""
    # Convert images to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray_frame, None)
    kp2, des2 = orb.detectAndCompute(gray_template, None)

    # Ensure descriptors are not None before matching
    if des1 is None or des2 is None:
        print("No descriptors found in one or both images.")
        return [], [], [], gray_frame

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in ascending order of distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches, kp1, kp2, gray_frame

def contour_length_or_area(contour):  
    """Calculate the length of a line/curve or the area of a filled shape."""  
    
    # Check if the contour is valid  
    if len(contour) == 0:  
        raise ValueError("The contour is empty.")  
    
    # Use the arcLength function to calculate the perimeter  
    length = cv2.arcLength(contour, closed=False)  
    
    # Use the contourArea function to calculate the area  
    area = cv2.contourArea(contour)  
    
    # Determine if the contour is closed (indicating a filled shape)  
    is_closed = True if cv2.isContourConvex(contour) else False  
    
    # Return length if it's likely a line or curve, otherwise return area  
    if is_closed:  
        return area  # If the contour is a closed shape (filled), return the area.  
    else:  
        return length  # If it's an open contour (line or curve), return the length.  

def process_image(image):  
    """Make everything white if it is not black, and make black lines heavier."""  
    
    # Convert image to grayscale  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    
    # Create a mask for black pixels: set a threshold to define "black"  
    # Here we consider pixels with intensity below 50 as black  
    black_mask = gray < 100  # Boolean mask for black pixels  
    
    # Create an output image initialized to white  
    result = np.ones_like(image) * 255  # Start with a white image  
    
    # Use morphological operations to thicken the black lines  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Define kernel for dilation  
    dilated_black = cv2.dilate(gray, kernel)  # Dilate the grayscale image to thicken the black areas  
    
    # Create a mask from the dilated image where we want to keep black  
    dilated_mask = dilated_black < 100  # Mask for dilated black areas  
    
    # Combine the masks to retain only the thickened black lines in the result  
    result[dilated_mask] = image[dilated_mask]  # Retain original color in black areas  
    
    return result

def draw_contours_with_numbers(frame, contours, numbers):  
    """  
    Draw contours on the frame and annotate them with numbers.  

    Parameters:  
    - frame: The image on which to draw the contours.  
    - contours: A list of contours to draw.  
    - numbers: A list of numbers to annotate the contours.  

    Returns:  
    - Annotated frame with drawn contours and numbers.  
    """  
    # Loop through the contours and numbers  
    for i, contour in enumerate(contours):  
        # Draw the contour  
        cv2.drawContours(frame, contours, i, (0, 255, 0), 2)  # Green color for contours  

        # Get the centroid of the contour for positioning the text  
        M = cv2.moments(contour)  
        if M["m00"] != 0:  # Avoid division by zero  
            cX = int(M["m10"] / M["m00"])  
            cY = int(M["m01"] / M["m00"])  
        else:  
            # If the contour is very small and moments can't be computed, use the first point  
            cX, cY = contour[0][0]  

        # Annotate the contour with the corresponding number  
        cv2.putText(frame, str(numbers[i]), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  

    return frame
def main():
    global last_template 
    cap = cv2.VideoCapture('http://192.168.1.8:8080/video')  # Use webcam (change to camera URL if necessary)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return
    last_template = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        frame = rescale(frame,0.7)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            last_template = similarity_detector(frame)
            print("Red sign detected")

        # If the template has been recorded, match it if 'p' is pressed
        if last_template is not None:
            if key == ord('p'):  # Search for matches only when 'p' is pressed
                nau = detect_red_and_blackout(frame)
                cv2.imshow("reds", nau)
                cnts = last_template(nau)
                # Draw matches
                length = contour_length_or_area(cnts)
                print(f"length is {length}")
                nau = process_image(frame)
                cv2.imshow("areas", nau)
                b = calculate_area(nau,length)
                for a in b:
                    print(a)
               

        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
