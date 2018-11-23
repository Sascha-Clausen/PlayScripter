import numpy as np
import cv2
cap = cv2.VideoCapture('1.mp4')

#bd = cv2.createLineSegmentDetector()
bd = cv2.line_descriptor.LSDDetector_createLSDDetector()


# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame_gray = cv2.Canny(frame_gray, 100, 100)

    #lines, width, prec, nfa = bd.detect(frame_gray)
    lines = bd.detect(frame_gray, 2, 2)

    #lines = lines[width > max(width) * 0.25]
    #lengths = np.array([cv2.norm(line[0][2:4] - line[0][0:2]) for line in lines])
    lengths = np.array([cv2.norm(np.array(line.getEndPoint()) - np.array(line.getStartPoint())) for line in lines])
    #lines = np.array(lines)[lengths > max(lengths) * 0.5]
    lines = np.array(lines)[lengths >  2 * np.mean(lengths)]

    for line in lines:
      #line = line.ravel()
      #frame = cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
      start_point = line.getStartPoint()
      start_point = (int(start_point[0]), int(start_point[1]))
      end_point = line.getEndPoint()
      end_point = (int(end_point[0]), int(end_point[1]))
      frame = cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(2) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()
