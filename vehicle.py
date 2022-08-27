import cv2
import numpy as np

# Web camera
cap = cv2.VideoCapture("traffic.mp4")

min_width_rect = 80  # Min width of rectangle
min_height_rect = 80  # Min height of rectange

count_line_position = 550

# Initializing subtractor (using algorithm)
# Subtract background from a image (focusing or detecting on a vehicle only)
algo = cv2.createBackgroundSubtractorMOG2(history=1000)


def center_finder(x, y, w, h):  # Returns the coordinate of center point of bounding rectangle
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy


vehicle_counter = []
"""
We will store coordinate value of center of bounding rectangle
and then append the center in the list, after that we use len()
function to find the length of list which keeps on increasing as
the vehicle passes across the blue line.
"""
offset = 5  # Allowable error between the pixel
counter = 0

while True:
    ret, frame1 = cap.read()

    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 11)
    # Applying on each frame
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)), iterations=1)
   # kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_ = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, (5, 5))
    dilate_ = cv2.morphologyEx(dilate_, cv2.MORPH_CLOSE, (5, 5))
    counter_shape, h = cv2.findContours(
        dilate_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cv2.line(frame1, (25, count_line_position),
             (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counter_shape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame1, "Vehicle" + str(counter), (x, y-20),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

        center = center_finder(x, y, w, h)
        vehicle_counter.append(center)
        # print(vehicle_counter)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in vehicle_counter:
            if y < (count_line_position + offset) and y > (count_line_position-offset):
                counter += 1
                vehicle_counter.remove((x, y))
                print(f"Vehicle counter:{counter}")

    cv2.putText(frame1, "Vehicle counter :" + str(counter),
                (450, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)

    #cv2.imshow("Traffic camera", dilate_)
    cv2.imshow("Real traffic video", frame1)
    # cv2.imshow("Real traffic video", counter_shape)

    if cv2.waitKey(50) == 13:  # 13 is a 'ASCII' code for escape character
        break

cv2.destroyAllWindows()
cap.release()
