import cv2  
import numpy as np  

# Function to calculate the length of the red line  
def measure_red_line(frame):  
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
            
    return max_length  

# Function to calculate areas of shapes detected  
def calculate_area(frame, scale):  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    areas = []  
    for cnt in contours:  
        area = cv2.contourArea(cnt)  
        if area > 1000:  # Threshold for minimum area  
            areas.append(area / (scale ** 2))  # Calculate the area relative to the scale  

            # Draw the detected shapes  
            cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)  

            # Calculate the centroid for displaying the area  
            M = cv2.moments(cnt)  
            if M['m00'] != 0:  
                cX = int(M['m10'] / M['m00'])  
                cY = int(M['m01'] / M['m00'])  
                cv2.putText(frame, f"Area: {area / (scale ** 2):.2f}", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  

    return areas  

# Main function  
def main():  
    cap = cv2.VideoCapture('http://192.168.1.6:8080/video')  

    print("Press 'r' to measure the red line for calibration.")  
    
    scale = None  # Will store the length of the red line when measured  

    while True:  
        ret, frame = cap.read()  
        if not ret:
            break  

        # If the scale has been measured, display it  
        if scale is not None:  
            cv2.putText(frame, f"Current Scale: {scale:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  

        cv2.imshow("Camera Feed - Red Line Measurement", frame)  

        key = cv2.waitKey(1) & 0xFF  
        if key == ord('r'):  # Measure and update the scale when 'r' is pressed  
            scale = measure_red_line(frame)  
            if scale > 0:  
                print(f"Measured Red Line Length: {scale:.2f} pixels")  
            else:  
                print("No red line detected. Please ensure it is visible.")  
        
        if key == ord('p'):  # When 'p' is pressed, switch to area calculation mode  
            print("Now draw shapes on the white page. Press 'p' to calculate areas.")  
            break  

        if key == ord('q'):  
            break  

    # Begin area calculation loop  
    all_areas = []  # List to store areas of shapes  

    while True:  
        ret, frame = cap.read()  
        if not ret:  
            break  

        cv2.imshow("Camera Feed - Area Calculation", frame)  

        key = cv2.waitKey(1) & 0xFF  
        if key == ord('p'):  # When 'p' is pressed, calculate areas  
            if scale is None:  
                print("Scale not yet determined. Please measure the red line first using 'r'.")  
            else:  
                areas = calculate_area(frame, scale)  
                all_areas.extend(areas)  # Add new areas to the list  

                # Display all calculated areas on the frame  
                for index, area in enumerate(all_areas):  
                    cv2.putText(frame, f"Shape {index + 1}: {area:.2f} units²", (10, 50 + index * 20),  
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  

                # Print areas to console  
                for index, area in enumerate(all_areas):  
                    print(f"Shape {index + 1} Area: {area:.2f} units²")  

        if key == ord('q'):  
            break  

    cap.release()  
    cv2.destroyAllWindows()  

if __name__ == "__main__":  
    main()