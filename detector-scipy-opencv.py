# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import glob
import cv2
from ctypes import *
import time

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    #boxes = dn.make_boxes(net)
    boxes = dn.make_network_boxes(net)
    probs = dn.make_probs(net)
    num =   dn.num_boxes(net)
    dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    #im = dn.load_image(image, 0, 0)
    #im = dn.load_image(bytes(image, encoding='utf-8'), 0, 0)
    im = image
    num = c_int(0)
    pnum = pointer(num)
    dn.predict_image(net, im)
    dets = dn.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    if isinstance(im, bytes):
      dn.free_image(im)
    dn.free_detections(dets, num)
    return res

def detect_opencv(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = array_to_image(image)
    dn.rgbgr_image(im)
    return detect(net, meta, im, thresh, hier_thresh, nms)

def get_corners_from_detection(detection):
    width =  int(detection[2][2])
    height = int(detection[2][3])
    left = int(detection[2][0] - (width / 2))     #center_x = p[2][0]
    top  = int(detection[2][1] - (height / 2)) #-10  #center_y = p[2][1]
    top_left = (left, top)
    right  = int(detection[2][0] + (width / 2))
    bottom = int(detection[2][1] + (height / 2))
    bottom_right = (right, bottom)
    return top_left, bottom_right

def draw_detections(image, detections):
    for detection in detections:
        top_left, bottom_right = get_corners_from_detection(detection)

        contained_detections = [det for det in detections if
                                get_corners_from_detection(det)[0][0] > top_left[0] and
                                get_corners_from_detection(det)[0][1] > top_left[1] and
                                get_corners_from_detection(det)[1][0] < bottom_right[0] and
                                get_corners_from_detection(det)[1][1] < bottom_right[1]]
        if len(contained_detections) > 0:
            print('Filter detection')
        else:
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    return image

def init():
    # Darknet
    net = dn.load_net(b"/data1/sclausen/darknet/cfg/yolov3.cfg", b"/data1/sclausen/darknet/yolov3.weights", 0)
    meta = dn.load_meta(b"/data1/sclausen/darknet/cfg/coco.data")

    return net, meta

def detect_all(net, meta):
    totalTime = 0

    images = sorted(glob.glob('output/*.jpg'))

    for i, img in enumerate(images[:5]):
        print('Image {} / {}'.format(i + 1, len(images)))
        cvImg = cv2.imread(img)
        startTime = time.process_time()
        r = detect_opencv(net, meta, cvImg, thresh=.25, nms=.5)
        endTime = time.process_time()
        totalTime += endTime - startTime

        detections = draw_detections(cvImg, r)
        cv2.imwrite('output/detected/{:04d}_detected.jpg'.format(i), detections)

        #print(r)
    print('Total Time: {}'.format(totalTime))

import sys, os
#sys.path.append(os.path.join(os.getcwd(),'python/'))
sys.path.append('/data1/sclausen/darknet/python')
sys.path.append('/usr/local/cuda-9.0/lib64')

import darknet as dn

"""
r = dn.detect(net, meta, "data/dog.jpg")
print r

# scipy
arr= imread('data/dog.jpg')
im = array_to_image(arr)
r = detect2(net, meta, im)
print r
"""

