import cv2  
import numpy as np  

def rescale(frame, scale: float = 0.8) -> np.ndarray:  
    # Get dimensions of the original frame  
    height, width = frame.shape[:2]  
    
    # Calculate new dimensions  
    new_dims = (int(width * scale), int(height * scale))  
    
    # Resize the frame
    resized_frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)  
    
    return resized_frame

def see_frame(frame):
    frame = rescale(frame)
    return frame

# Function to calculate the length of the red line  
def measure_red_line(cap):  
    final_size = 0
    for i in range(100):
        ret, frame = cap.read()
        frame = see_frame(frame)
        # Convert the frame to HSV color space  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  

        # Define the range for the color red  
        lower_red1 = np.array([0, 100, 100])  
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])  
        upper_red2 = np.array([180, 255, 255])  

        # Create masks for red color  
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  
        mask = mask1 + mask2

        # Find contours in the mask  
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        
        max_length = 0  
        for cnt in contours:
            length = cv2.arcLength(cnt, True)  # Calculate the length of the contour
            if length > max_length:  # Keep track of the maximum length found  
                max_length = length  
                
        final_size += max_length
    return final_size / 100

# Function to calculate areas of shapes detected  
def calculate_area(frame, scale):  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    areas = []  
    cnts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Threshold for minimum area  
            areas.append(area / (scale ** 2) * 100)  # Calculate the area relative to the scale
            cnts.append(cnt)
    return areas, tuple(cnts)

# Main function  
def main():  
    cap = cv2.VideoCapture('http://192.168.1.8:8080/video')

    cv2.namedWindow("Calculate Areas")  # Create a single named window called ""Calculate Areas"
    print("Press 'r' to measure the red line for calibration.")  
    
    scale = None  # Will store the length of the red line when measured

    all_areas = []

    shape_index = 0

    contours = []

    while True:
        ret, frame = cap.read()
        frame = see_frame(frame)

        
        if not ret:
            break

        if scale is None:
            cv2.putText(frame, f"press 'r' to calibrate the scale", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            if not all_areas:
                cv2.putText(frame, f"Current Scale: {scale:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"press 'p' to calculate the areas", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                #cv2.putText(frame, f"Shape index: {shape_index}. Area: {all_areas[shape_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                #cv2.putText(frame, f"press 'r' to recalibrate the scale", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                for i in range(len(all_areas)):
                    # Draw the detected shapes  
                    cv2.drawContours(frame, [contours[i]], -1, (0, 0, 255), 2)  

                    # Calculate the centroid for displaying the area  
                    M = cv2.moments(contours[i])
                    if M['m00'] != 0:
                        cX = int(M['m10'] / M['m00'])
                        cY = int(M['m01'] / M['m00'])
                        cv2.putText(frame, f"Area: {all_areas[i]}", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  

        key = cv2.waitKey(1) & 0xFF  
        if key == ord('r'):  # Measure and update the scale when 'r' is pressed
            scale = measure_red_line(cap)
            all_areas = []
        
        if key == ord('p'):  # When 'p' is pressed, switch to area calculation mode  
            if scale is not None:
                all_areas, contours = calculate_area(frame, scale)

        if key == ord('q'):
            break  

        cv2.imshow("Calculate Areas", frame)  # Show all frames in the ""Calculate Areas" window

    cap.release()  
    cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()