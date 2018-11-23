import numpy as np
import cv2 as cv

cap = cv.VideoCapture('1.mp4')
# params for ShiTomasi corner detection
maxCorners = 1000
feature_params = dict( maxCorners = maxCorners,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 3 )

# Create some random colors
color = np.random.randint(0,255,(maxCorners,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
points = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
  ret,frame = cap.read()
  frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  points = cv.goodFeaturesToTrack(cv.Canny(frame_gray,100,200), mask = None, **feature_params)
  # draw the points
  for i, point in enumerate(points):
      a,b = point.ravel()
      frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
  cv.imshow('frame',frame)
  k = cv.waitKey(30) & 0xff
  if k == 27:
      break
cv.destroyAllWindows()
cap.release()
