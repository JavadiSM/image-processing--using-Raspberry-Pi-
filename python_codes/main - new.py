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

def main():  
    cap = cv2.VideoCapture('http://192.168.100.12:8080/video')

    cv2.namedWindow("Calculate Areas")
    print("Press 'r' to measure the red shape for calibration.")  
    
    scale = None
    scale_contour = None

    all_areas = []
    contours = []
    selected_shape_index = 0

    # Initial program state: 0 = initial, 1 = scale calculation, 2 = area calculation
    program_state = 0

    while True:
        ret, frame = cap.read()
        frame = rescale(frame)
        backup_frame = frame.copy()

        if not ret:
            break

        if program_state == 0:  # Initial state
            cv2.rectangle(frame, (5, 0), (600, 45), (0, 0, 0), -1); cv2.putText(frame, f"Press 'r' to calibrate the scale, press 'q' to exit", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        elif program_state == 1:  # Scale calculation state
            if scale_contour is not None:
                cv2.drawContours(frame, [scale_contour], -1, (0, 0, 255), 2)

                M = cv2.moments(scale_contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])

                    # Display scale
                    cv2.putText(frame, f"Scale: {scale:.2f} px", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (5, 0), (905, 45), (0, 0, 0), -1); cv2.putText(frame, f"Press 'p' to calculate the areas, press 'r' to recalibrate, press 'q' to exit", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        elif program_state == 2:  # Area calculation state
            if scale is not None:
                cv2.drawContours(frame, [contours[selected_shape_index]], -1, (0, 0, 255), 2)

                M = cv2.moments(contours[selected_shape_index])
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    cv2.putText(frame, f"Area: {all_areas[selected_shape_index]:.2f}", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.rectangle(frame, (5, 0), (500, 45), (0, 0, 0), -1); cv2.putText(frame, f"Press 'r' to recalibrate, press 'q' to exit", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (5, 35), (440, 75), (0, 0, 0), -1); cv2.putText(frame, f"Showing shape {selected_shape_index+1} out of {len(all_areas)} shapes", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        key = cv2.waitKey(1) & 0xFF  

        # Key handling based on program state
        if key == ord('r'):
            program_state = 1
            scale_frame = backup_frame.copy()
            scale_frame = process_scale_frame(scale_frame)
            scale, scale_contour = calculate_scale(scale_frame)
            all_areas = []

        elif key == ord('p'):
            if program_state == 1 or program_state == 2:
                program_state = 2
                calculation_frame = backup_frame.copy()
                calculation_frame = process_area_frame(calculation_frame)
                all_areas, contours = calculate_area(calculation_frame, scale)

                # sort shapes by x-coordinate
                cX_values = []
                for contour in contours:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cX = int(M['m10'] / M['m00'])
                    else:
                        cX = 0
                    cX_values.append(cX)

                sorted_data = sorted(zip(cX_values, contours, all_areas), key=lambda x: x[0])

                cX_values, contours, all_areas = zip(*sorted_data)

                contours = list(contours)
                all_areas = list(all_areas)

        elif key == ord('a') or key == ord('d'):
            if program_state == 2:
                selected_shape_index -= 1 if key == ord('a') else -1
                selected_shape_index = selected_shape_index % len(all_areas)

        elif key == ord('q'):
            break  

        cv2.imshow("Calculate Areas", frame)

    cap.release()  
    cv2.destroyAllWindows()

if __name__ == "__main__":  
    main()
