from __future__ import print_function

import os,sys
import math
import time

import cv2
import numpy as np

# set this to your path
detector_dir = '../face-datasets/'
sys.path.insert(0, detector_dir+'facealign')
sys.path.insert(0, detector_dir+'util')

from MtcnnPycaffe import MtcnnDetector
import PyLandmark as LandmarkDetector

def point_p2i(p):
    shape  =[]
    for k in range(int(len(p)/2)):
        shape.append(int(p[k]))
        shape.append(int(p[k+5]))
    return shape

    
def draw_rect(img, bbox):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
          (int(bbox[2]),int(bbox[3])) ,(0,0,255), 2)

          
def draw_landmark(img, landmarks):
    for i in range(len(landmarks)/2):
        cv2.circle(img,(int(round(landmarks[i*2])),int(round(landmarks[i*2+1]))),1,(0,255,0),2)
       
       
def draw_pose(img, bbox, pose):
    text = '[%.0f,%.0f,%.0f]' % (pose[0], pose[1], pose[2])
    cv2.putText(img, text, (int(bbox[0]), int(bbox[1]-10)), 0, 0.5, (0,255,0), 2)
    
            
if __name__ == '__main__':
    img_path = sys.argv[1]
    im = cv2.imread(img_path)
    # create detector
    detector = MtcnnDetector()
    bboxes, points = detector.detect_face(im)
    poses = []
    landmarks = []
    # set datadir
    LandmarkDetector.create("./model/")
    # faces
    for i in range(len(bboxes)):
        rect = [int(bboxes[i][0]), int(bboxes[i][1]), 
          int(bboxes[i][2]-bboxes[i][0]), int(bboxes[i][3]-bboxes[i][1])]
        pts = LandmarkDetector.detect(im, rect, point_p2i(points[i]), 0)
        #pts = LandmarkDetector.detect(im, rect, [], 1)
        pose = LandmarkDetector.getPose(pts)
        landmarks.append(pts)
        poses.append(pose)
        #print(pts)
    LandmarkDetector.destroy()
    # show
    bboxes = bboxes.tolist()
    for i in range(len(bboxes)):
        draw_rect(im, bboxes[i])
        draw_landmark(im, landmarks[i])
        draw_pose(im, bboxes[i], poses[i])
        
    cv2.imwrite(img_path+'.detect.jpg', im)

