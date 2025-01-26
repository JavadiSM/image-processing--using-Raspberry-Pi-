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
    return areas, tuple(cnts)

def find_red_objects(frame):
    """
    Process the frame to find red objects and return the mask.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the range for red color
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to improve the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
def rescale(frame, scale: float = 0.8) -> np.ndarray:  
    # Get dimensions of the original frame  
    height, width = frame.shape[:2]  
    
    # Calculate new dimensions  
    new_dims = (int(width * scale), int(height * scale))  
    
    # Resize the frame
    resized_frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)  
    
    return resized_frame
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

def main():
    global last_template
    cap = cv2.VideoCapture('http://192.168.1.8:8080/video')  # Use webcam (change to camera URL if necessary)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        frame = rescale(frame,0.2)
        cv2.imshow('Frame', frame)
        mask = find_red_objects(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        key = cv2.waitKey(1) & 0xFF

        red_object_area = None  # To store the area of the red object

        # Draw all found contours
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

                # Extract the region of interest (ROI) as a template when 'r' is pressed
                if last_template is None:  # Only capture the template after detecting an object
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if key == ord('r'):
                        last_template = frame[y:y + h, x:x + w]  # Capture the detected red sign
                        red_object_area = cv2.contourArea(contour)
                        print("Red sign detected")

        # If the template has been recorded, match it if 'p' is pressed
        if last_template is not None:
            if key == ord('p'):  # Search for matches only when 'p' is pressed
                matches, kp1, kp2, gray_frame = feature_based_matching(frame, last_template)

                # Draw matches
                for match in matches[:10]:  # Draw only the first 10 matches for better visualization
                    cv2.circle(frame, (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1])), 5, (255, 0, 0), -1)

                # Display matched keypoints in the frame and template
                cv2.imshow("Matches", frame)

                # Calculate arc length (perimeter) of the matched object
                if len(matches) > 0:  # Ensure there are enough matches
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h, w = last_template.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                        # Calculate arc length (perimeter) of the matched object
                        arc_length = cv2.arcLength(np.int32(dst), True)  # Closed shape
                        print(f"Matched object arc length: {arc_length}")

                        # Optionally, draw the arc length on the frame
                        x, y, _, _ = cv2.boundingRect(np.int32(dst))
                        cv2.putText(frame, f"Length: {arc_length:.2f} px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


               

        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
