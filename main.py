import math

import cv2 as cv
import numpy as np

cv.namedWindow('trackbars')
source = cv.VideoCapture(0)

min_hsv = np.array([17, 92, 108])
max_hsv = np.array([47, 255, 255])
focus_length = 545
width = 0.2
frame_width = 640


def callback(_):
    pass


def calculate_distance(p):
    d = width * focus_length / p
    return d


def calculate_ratio():
    ratio = width / 640
    height2 = 100 * ratio
    return height2


def calculate_alpha(height2):
    j = 320
    a = height2 ** 2 + j ** 2
    b = math.sqrt(a)
    alpha = 61
    return alpha


def calculate_angle():
    pass


cv.createTrackbar('min_h', 'trackbars', min_hsv[0], 180, callback)
cv.createTrackbar('min_s', 'trackbars', min_hsv[1], 255, callback)
cv.createTrackbar('min_v', 'trackbars', min_hsv[2], 255, callback)
cv.createTrackbar('max_h', 'trackbars', max_hsv[0], 180, callback)
cv.createTrackbar('max_s', 'trackbars', max_hsv[1], 255, callback)
cv.createTrackbar('max_v', 'trackbars', max_hsv[2], 255, callback)

while cv.waitKey(1) & 0xFF != 27:
    frame_exists, frame = source.read()
    # print(type(frame_exists))

    # if not frame_exists:
    #     print("bye")
    #     break

    if frame_exists:
        frame = cv.GaussianBlur(frame, (5, 5), 0)
        ksize = (10, 10)
        frame = cv.blur(frame, ksize)

        min_hsv[0] = cv.getTrackbarPos('min_h', 'trackbars')
        min_hsv[1] = cv.getTrackbarPos('min_s', 'trackbars')
        min_hsv[2] = cv.getTrackbarPos('min_v', 'trackbars')

        max_hsv[0] = cv.getTrackbarPos('max_h', 'trackbars')
        max_hsv[1] = cv.getTrackbarPos('max_s', 'trackbars')
        max_hsv[2] = cv.getTrackbarPos('max_v', 'trackbars')

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_frame = cv.inRange(hsv_frame, min_hsv, max_hsv)

        hsv_frame = cv.erode(hsv_frame, (5, 5))
        hsv_frame = cv.dilate(hsv_frame, (5, 5))

        cnts, h = cv.findContours(hsv_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        hulls = []
        maxValue = 5
        maxHull = None
        for i, cnt in enumerate(cnts):
            hull = cv.convexHull(cnt)
            area = cv.contourArea(cnt)
            hull_area = cv.contourArea(hull)
            if hull_area != 0:
                solidity = float(area) / hull_area
                if solidity > 0.5:
                    hulls.append(hull)
                    hulls.append(cnts)
                    if hull_area > maxValue:
                        maxValue = hull_area
                        maxHull = hull

                # print("solidity: ", solidity)
            else:
                approximation = cv.approxPolyDP(cnts[i], cv.arcLength(cnts[i], True) * 0.01, True)
                # print(len(approximation))
        if len(hulls) != 0 and maxHull is not None:
            cv.drawContours(frame, [maxHull], -1, (255, 0, 0), 4)
            x, y, w, h = cv.boundingRect(maxHull)
            distance = calculate_distance(w)

        frame = cv.bitwise_and(frame, frame, mask=hsv_frame)
        if maxHull is not None:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            # cv.putText(frame, 'Distance= ' + str(distance), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
            #            cv.LINE_4)
            cv.putText(frame,
                       'ANGLE= ' + str(np.interp(x + w // 2, [0, 640], [-30.5, 30.5])), (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_4)
        cv.imshow('images', frame)

source.release()
cv.destroyAllWindows()
