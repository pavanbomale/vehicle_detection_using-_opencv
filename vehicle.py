import cv2 as cv
import numpy as np

#detects vehicles that are minwidth and minheight of 80
rect_minwidth = 80
rect_minHeight = 80

#load video
capture = cv.VideoCapture('video.mp4')

#center point of rectangle
def rect_center(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx, cy

#holds the vehicles detected
detected = []
counter = 0
offset =6

# subtractor algorithm from cv2
algorithm = cv.bgsegm.createBackgroundSubtractorMOG()

#run video until keyboard interrupt
while True:
    ret, video_frame = capture.read()
    setGrey = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)
    setBlur = cv.GaussianBlur(setGrey, (3,3), 5)
    #apply algorithm on each frame
    img_sub = algorithm.apply(setBlur)
    dilate = cv.dilate(img_sub, np.ones((5,5)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    dilate2 = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
    dilate2 = cv.morphologyEx(dilate2, cv.MORPH_CLOSE, kernel)
    counter_shape, h = cv.findContours(dilate2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow('Vehicle Detection using python', dilate2)

    #drawing crossing line
    cv.line(video_frame , (25,550), (1200, 550), (255,0,0), 3)

    for (i, c) in enumerate(counter_shape):
        (x,y,w,h) = cv.boundingRect(c)
        validate_counter = (w >= rect_minwidth) and (h >= rect_minHeight)
        if not validate_counter:
            continue
        cv.rectangle(video_frame, (x,y), (x+w, y+h), (0,255,0), 3)
        center = rect_center(x,y,w,h)
        detected.append(center)
        cv.circle(video_frame, center, 4, (0,0,255), -1)

        for (x,y) in detected :
            #if the vehicle crossed the blue line (550)
            if y<(550+offset) and y>(550-offset):
                counter += 1
                cv.line(video_frame , (25,550), (1200, 550), (255, 255,0), 3)
                detected.remove((x,y))

    cv.putText(video_frame, 'vehicles crossed :' + str(counter), (450, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255,0,125), 4)

    #window
    cv.imshow('Vehicle Detection using python', video_frame)
    if cv.waitKey(9) == 13: #hitting enter key
        break
    
cv.destroyAllWindows()
capture.release()