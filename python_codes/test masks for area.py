import cv2
import numpy as np

def rescale(frame, scale: float = 0.8) -> np.ndarray:  
    height, width = frame.shape[:2]  
    new_dims = (int(width * scale), int(height * scale))  
    resized_frame = cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)  
    return resized_frame

def main():
    # Initialize webcam feed
    cap = cv2.VideoCapture('http://192.168.50.181:8080/video')

    # Initial parameters
    canny_low = 50  # Lower threshold for Canny
    canny_high = 150  # Upper threshold for Canny
    kernel_size = 5  # Kernel size for filtering

    print("Controls:")
    print("1/2 - Decrease/Increase Canny Low Threshold")
    print("3/4 - Decrease/Increase Canny High Threshold")
    print("5/6 - Decrease/Increase Kernel Size")
    print("q - Quit")

    while True:
        ret, frame = cap.read()
        frame = rescale(frame)
        if not ret:
            print("Failed to grab frame. Check your webcam feed.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filtering to preserve edges and reduce noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Detect edges using Canny
        edges = cv2.Canny(filtered, canny_low, canny_high)

        # Morphological operations to refine edges
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges_cleaned = cv2.dilate(edges, kernel, iterations=1)
        edges_cleaned = cv2.erode(edges_cleaned, kernel, iterations=1)

        # Display the edges
        cv2.putText(edges_cleaned, f"Canny Low: {canny_low}, High: {canny_high}, Kernel: {kernel_size}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.imshow("Edges", edges_cleaned)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):  # Decrease Canny low threshold
            canny_low = max(0, canny_low - 5)
        elif key == ord('2'):  # Increase Canny low threshold
            canny_low = min(255, canny_low + 5)
        elif key == ord('3'):  # Decrease Canny high threshold
            canny_high = max(0, canny_high - 5)
        elif key == ord('4'):  # Increase Canny high threshold
            canny_high = min(255, canny_high + 5)
        elif key == ord('5'):  # Decrease kernel size
            kernel_size = max(3, kernel_size - 2)
        elif key == ord('6'):  # Increase kernel size
            kernel_size = kernel_size + 2
        elif key == ord('q'):  # Quit the loop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()