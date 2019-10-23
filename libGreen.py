import sys
import os, time
import glob
import cv2
import imutils
import numpy as np
import math
from sklearn.externals import joblib
import _pickle as cPickle
from scipy.spatial import distance
from random import randint

class webCam:
    def __init__(self, id=0, videofile="", size=(1920, 1080)):
        self.camsize = size
        #for FPS count
        self.start_time = time.time()
        self.last_time = time.time()
        self.total_frames = 0
        self.last_frames = 0
        self.fps = 0
        self.out = None

        if(len(videofile)>0):
            self.cam = cv2.VideoCapture(videofile)
            self.playvideo = True

            camera = self.cam
            self.video_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        else:
            self.cam = cv2.VideoCapture(id)
            #self.cam = cv2.VideoCapture(cv2.CAP_DSHOW+id)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.playvideo = False

    def set_record(self, outputfile="output.avi", video_rate=25):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(outputfile, fourcc, video_rate, self.video_size)
        self.out = out

    def write_video(self, frame):
        out = self.out
        out.write(frame)
        self.out = out

    def stop_record(self):
        out = self.out
        out.release()
        self.out = out

    def fps_count(self, seconds_fps=10):
        fps = self.fps

        timenow = time.time()
        if(timenow - self.last_time)>seconds_fps:
            fps  = (self.total_frames - self.last_frames) / (timenow - self.last_time)
            self.last_frames = self.total_frames
            self.last_time = timenow
            self.fps = fps

        return round(fps,2)

    def working(self):
        webCam = self.cam
        if(webCam.isOpened() is True):
            return True
        else:
            if(self.playvideo is True):
                return True
            else:
                return False

    def camRealSize(self):
        webcam = self.cam
        width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def getFrame(self, rotate=0, vflip=False, hflip=False, resize=None):
        webcam = self.cam
        hasFrame, frame = webcam.read()
        if(frame is not None):
            if(vflip==True):
                frame = cv2.flip(frame, 0)
            if(hflip==True):
                frame = cv2.flip(frame, 1)
    
            if(rotate>0):
                frame = imutils.rotate_bound(frame, rotate)
            if(resize is not None):
                frame_resized = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
            else:
                frame_resized = None

        else:
            frame = None
            hasFrame = False
            frame_resized = None

        self.total_frames += 1


        return hasFrame, frame_resized, frame

    def release(self):
        webcam = self.cam
        webcam.release()

class OBJTracking:
    def __init__(self):
        self.multiTracker = cv2.MultiTracker_create()
        self.roi_bboxes = []
        self.roi_colors = []

    def create_tracker(self, trackerType):
        trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

        # Create a tracker based on tracker name
        if trackerType == trackerTypes[0]:
            tracker = cv2.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]:
            tracker = cv2.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
            tracker = cv2.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
            tracker = cv2.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
            tracker = cv2.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
            tracker = cv2.TrackerGOTURN_create()
        elif trackerType == trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == trackerTypes[7]:
            tracker = cv2.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
                print(t)

        return tracker

    def setROIs(self, frame, bboxes, trackerType):
        multiTracker = self.multiTracker 

        ## Select boxes
        self.roi_bboxes = []
        self.roi_colors = []

        for i, bbox in enumerate(bboxes):
            self.roi_bboxes.append(bbox)
            self.roi_colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))

            multiTracker.add(self.create_tracker(trackerType), frame, bbox)

        self.multiTracker = multiTracker

    def trackROI(self, frame):
        multiTracker = self.multiTracker

        success, boxes = multiTracker.update(frame)
        self.multiTracker = multiTracker

        return (success, boxes)
