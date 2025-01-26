import cv2
import numpy as np

# Global variables
calibration_done = False
reference_contour = None
reference_area = None
scale_factor = 1.0  # Pixels to real-world scale

def find_red_objects(frame):
    """
    Detect red objects in the frame and return the mask and contours.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

def calculate_area(contour):
    """
    Calculate the area of a given contour in real-world units.
    """
    area_in_pixels = cv2.contourArea(contour)
    area_in_real_world = area_in_pixels * scale_factor
    return area_in_real_world

def main():
    global calibration_done, reference_contour, reference_area, scale_factor

    cap = cv2.VideoCapture('http://192.168.1.8:8080/video')  # Open webcam
    if not cap.isOpened():
        print("Error: Cannot access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame.")
            break

        key = cv2.waitKey(1) & 0xFF

        if not calibration_done:
            # Calibration step: Find red objects and select one as the reference
            mask, contours = find_red_objects(frame)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # Ignore small noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'c' to calibrate", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if key == ord('c') and len(contours) > 0:
                # Select the largest contour as the reference
                reference_contour = max(contours, key=cv2.contourArea)
                reference_area = cv2.contourArea(reference_contour)
                known_size = 10.0  # Known size of the reference object (e.g., in cm^2)
                scale_factor = known_size / reference_area
                calibration_done = True
                print(f"Calibration done. Reference area: {reference_area}, Scale factor: {scale_factor}")
                cv2.destroyWindow("Calibration")
        else:
            # Detect all contours and calculate their areas relative to the reference
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the reference contour
            if reference_contour is not None:
                cv2.drawContours(frame, [reference_contour], -1, (0, 0, 255), 3)
                cv2.putText(frame, "Reference", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Process other contours
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # Ignore small noise
                    real_area = calculate_area(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Area: {real_area:.2f} cm^2", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Shapes", frame)

        if key == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
