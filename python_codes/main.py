import cv2  
import numpy as np
from functions import *

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
