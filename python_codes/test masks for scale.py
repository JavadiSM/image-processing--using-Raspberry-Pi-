import cv2
import numpy as np

def process_red_edges(frame, canny_low, canny_high, kernel_size):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red in HSV
    lower_red1 = np.array([0, 120, 70])  # Red hue part 1
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])  # Red hue part 2
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the original frame
    red_areas = cv2.bitwise_and(frame, frame, mask=red_mask)

    # Convert to grayscale to prepare for edge detection
    gray = cv2.cvtColor(red_areas, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Refine edges with morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges_cleaned = cv2.dilate(edges, kernel, iterations=1)
    edges_cleaned = cv2.erode(edges_cleaned, kernel, iterations=1)

    return edges_cleaned


def main():  
    cap = cv2.VideoCapture('http://192.168.50.181:8080/video')

    canny_low = 50
    canny_high = 150
    kernel_size = 3

    print("Controls:")
    print("1/2: Decrease/Increase Canny Low Threshold")
    print("3/4: Decrease/Increase Canny High Threshold")
    print("5/6: Decrease/Increase Kernel Size")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with current parameter values
        processed_frame = process_red_edges(frame, canny_low, canny_high, kernel_size)

        # Display current parameter values on the screen
        text = f"Canny Low: {canny_low}, Canny High: {canny_high}, Kernel Size: {kernel_size}"
        cv2.putText(processed_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Calculate Areas", processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit the program
            break
        elif key == ord('1'):  # Decrease canny_low
            canny_low = max(0, canny_low - 5)
        elif key == ord('2'):  # Increase canny_low
            canny_low = min(255, canny_low + 5)
        elif key == ord('3'):  # Decrease canny_high
            canny_high = max(0, canny_high - 5)
        elif key == ord('4'):  # Increase canny_high
            canny_high = min(255, canny_high + 5)
        elif key == ord('5'):  # Decrease kernel_size
            kernel_size = max(1, kernel_size - 2)  # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size -= 1
        elif key == ord('6'):  # Increase kernel_size
            kernel_size += 2  # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

    cap.release()  
    cv2.destroyAllWindows()


if __name__ == "__main__":  
    main()