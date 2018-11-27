import numpy as np
import cv2
cap = cv2.VideoCapture('1.mp4')

#sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(1000)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)

srcPoints = [(542, 151), (1029, 162), (604, 477), (372, 313)]
dstPoints = [(1066, 100), (1066, 500), (180, 500), (400, 300)]

# TODO:
# - filter keypoints on humans
# - filter keypoints outside of the rectified image

counter = 0
while(1):
  if counter > 0:
    ret, newFrame = cap.read()
    newGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    newKps, newDes = surf.detectAndCompute(frame, None)

    matches = flann.knnMatch(des, newDes, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
      if m.distance < 0.7 * n.distance:
        good.append(m)
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)
    matchesImg = cv2.drawMatches(frame, kps, newFrame, newKps, good, None, **draw_params)
    """cv2.imshow('Matches', matchesImg)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break"""

    srcPoints = np.float32([newKps[m.queryIdx].pt   for m in good])
    dstPoints = np.float32([kpDstCoords[m.trainIdx] for m in good])


  ret, frame = cap.read()

  M, mask = cv2.findHomography(np.float32(srcPoints), np.float32(dstPoints), cv2.RANSAC, ransacReprojThreshold=5.0)
  rectified = cv2.warpPerspective(frame, M, (1166, 600))

  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #kp = sift.detect(frame_gray, None)
  kps, des = surf.detectAndCompute(frame, None)
  #img = cv2.drawKeypoints(frame_gray, kps, frame)
  #cv2.imwrite('output/{:04d}.jpg'.format(counter), frame)
  kpSrcCoords = np.float32([kp.pt for kp in kps])
  kpDstCoords = cv2.perspectiveTransform(kpSrcCoords.reshape(-1,1,2), M)
  kpDstCoords = kpDstCoords.reshape(-1, 2)
  #for coords in kpDstCoords:
  #  cv2.circle(rectified, tuple(coords), 1, (0, 0, 255))

  counter += 1

  cv2.imshow('rectified',rectified)
  #cv2.imshow('frame',img)
  k = cv2.waitKey(2) & 0xff
  if k == 27:
    break

cv2.destroyAllWindows()
cap.release()
