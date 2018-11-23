import numpy as np
import cv2
cap = cv2.VideoCapture('1.mp4')

#sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(400)

# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
counter = 0
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #kp = sift.detect(frame_gray, None)
    #kp, des = surf.detectAndCompute(frame, None)
    #img = cv2.drawKeypoints(frame_gray, kp, frame)
    cv2.imwrite('output/{:04d}.jpg'.format(counter), frame)
    counter += 1

    cv2.imshow('frame',frame)
    k = cv2.waitKey(2) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
