#!/usr/bin/env python
import cv2
import numpy as np
import math
import time
import os
import subprocess
import threading
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
#from PIL import Image
import imutils
import roslib
#roslib.load_manifest('my_package')
import sys
import rospy
#from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#For ROS Subscribe
from sensor_msgs.msg import Imu
# from ..common import anorm2, draw_str

ASUS = True
STICK = False


class Detection:
    def __init__(self):
        #Define Variable###################################################################################
        self.save_img        = 1        # 0 : Don't save, 1: Save
        self.imu_sinc        = 13       # IMU time sinc

        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_114F0B00-video-index0"  # left 90 / 2 (board pose)
        # self.cam1_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_14480200-video-index0"  # left 0 / 1 (board pose)
        self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_2B460000-video-index0"  # right 45 / 3 (board pose)

        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_32490400-video-index0"  # right 0 / 5 (board pose)
        # self.cam1_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_1A480200-video-index0"  # left 45 / 4 (board pose)
        self.cam1_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_1A080800-video-index0"  # right 90 / 6 (board pose)



        ###################################################################################################

        #Define list
        self.ImgWidth        = 1920 # Resolution 
        self.ImgHeight       = 1080 # Resolution
        self.ImgFps          = 30   # Should be modified with fps
        self.sh_size         = (640,480)
        self.imu_data        = [[0]*4 for i in range(4)]
        self.imu_save        = [[0]*4 for i in range(self.imu_sinc)]
        self.R2D             = 180.0/np.pi
        self.D2R             = np.pi/180.0
        self.prev_time       = 0
        self.fps             = 0

        self.detect_interval = 1
        self.track_len       = 4
        self.tracks0         = []
        self.tracks1         = []
        self.tracks2         = []
        self.frame_idx0      = 0
        self.frame_idx1      = 0
        self.frame_idx2      = 0

        # Create some random colors
        self.color           = np.random.randint(0,255,(100,3))
        self.OF_CAM0_count   = 0
        self.OF_CAM1_count   = 0
        self.OF_CAM2_count   = 0

        self.Prev_CMO_img0 = []
        self.Prev_CMO_img1 = []
        self.Prev_CMO_img2 = []

        # Variables for LOG
        # self.

    def draw_str(self, dst, target, s):
        x, y = target
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
        cv2.putText(dst, s, (x, y),     cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    def saveImg(self, name, img):
        #img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(name, img)

    def img_processing(self, img, CAM_num):
        
        ### Resize img
        try :
            Shrink_img  = cv2.resize(img, self.sh_size, interpolation=cv2.IMREAD_COLOR)
        except :
            print("error in resize of CAM : " + str(CAM_num) )

        ### Rotate img
        if   CAM_num==0:
            Rot_img     =  imutils.rotate(Shrink_img, -1*self.imu_save[0][0])    # roll
        elif CAM_num==1:
            Rot_img     =  imutils.rotate(Shrink_img, -1*self.imu_save[0][3])    # ROT 45
        elif CAM_num==2:
            Rot_img     =  imutils.rotate(Shrink_img,    self.imu_save[0][1])    # pitch

        ### Grayscale
        Gray_img    = cv2.cvtColor(Rot_img, cv2.COLOR_BGR2GRAY)

        ### CMO
        kernel      = np.ones((5, 5), np.uint8)
        Open_img    = cv2.morphologyEx(Gray_img, cv2.MORPH_OPEN, kernel)
        Close_img   = cv2.morphologyEx(Gray_img, cv2.MORPH_CLOSE, kernel)
        CMO_img     = Close_img - Open_img

        img = CMO_img

        ### ptical Flow
        lk_params       = dict( winSize         = (15, 15),
                                maxLevel        = 2,
                                criteria        = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        feature_params  = dict( maxCorners      = 50,
                                qualityLevel    = 0.3,
                                minDistance     = 7,
                                blockSize       = 7 )

        if CAM_num==0:
            if  self.OF_CAM0_count==0:
                self.OF_CAM0_count = 1
                Prev_CMO_img = img.copy()
            else :
                Prev_CMO_img = self.Prev_CMO_img0.copy()
            TRACKS          = self.tracks0
            FRAME_IDX       = self.frame_idx0

        elif CAM_num==1:
            if  self.OF_CAM1_count==0:
                self.OF_CAM1_count = 1
                Prev_CMO_img = img.copy()
            else :
                Prev_CMO_img = self.Prev_CMO_img1.copy()
            TRACKS          = self.tracks1
            FRAME_IDX       = self.frame_idx1

        elif CAM_num==2:
            if  self.OF_CAM2_count==0:
                self.OF_CAM2_count = 1
                Prev_CMO_img = img.copy()
            else :
                Prev_CMO_img = self.Prev_CMO_img2.copy()
            TRACKS          = self.tracks2
            FRAME_IDX       = self.frame_idx2

        # Change gray_scale to Color for optical-flow-print
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if len(TRACKS) > 0:
            img0, img1      = Prev_CMO_img, CMO_img
            p0              = np.float32([tr[-1] for tr in TRACKS]).reshape(-1, 1, 2)
            p1,  _st, _err  = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _st, _err  = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d               = abs(p0-p0r).reshape(-1, 2).max(-1)
            good            = d < 1
            new_tracks      = []

            for tr, (x, y), good_flag in zip(TRACKS, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > self.track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            TRACKS = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in TRACKS], False, (0, 255, 0))
            # self.draw_str(vis, (20, 20), 'track count: %d' % len(TRACKS))

        if FRAME_IDX % self.detect_interval == 0:
            mask    = np.zeros_like(CMO_img)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in TRACKS]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p  = cv2.goodFeaturesToTrack(CMO_img, mask=mask, **feature_params)

            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    TRACKS.append([(x, y)])
        FRAME_IDX += 1

        if CAM_num==0:
            self.Prev_CMO_img0  = CMO_img
            self.tracks0        = TRACKS
            self.frame_idx0     = FRAME_IDX 
        elif CAM_num==1:
            self.Prev_CMO_img1  = CMO_img
            self.tracks1        = TRACKS
            self.frame_idx1     = FRAME_IDX 
        elif CAM_num==2:
            self.Prev_CMO_img2  = CMO_img
            self.tracks2        = TRACKS
            self.frame_idx2     = FRAME_IDX
        img = vis
        return img

    def QuaterniontoEuler(self ):
        # roll (x-axis rotation) calculation
        t0 = 2.0       * (self.imu_data[0][3] * self.imu_data[0][0] + self.imu_data[0][1] * self.imu_data[0][2])
        t1 = 1.0 - 2.0 * (self.imu_data[0][0] * self.imu_data[0][0] + self.imu_data[0][1] * self.imu_data[0][1])
        roll = math.atan2(t0,t1)*self.R2D

        # pitch ( y-axis rotation) calculation
        t2 = 2.0       * (self.imu_data[0][3] * self.imu_data[0][1] - self.imu_data[0][2] * self.imu_data[0][0])
        if   t2 > 1.0  :  t2 =  1.0
        elif t2 < -1.0 :  t2 = -1.0
        else           :  t2 =  t2
        pitch = - math.asin(t2)*self.R2D

        # yaw calculation
        t3 = 2.0       * (self.imu_data[0][3] * self.imu_data[0][2] + self.imu_data[0][0] * self.imu_data[0][1])
        t4 = 1.0 - 2.0 * (self.imu_data[0][1] * self.imu_data[0][1] + self.imu_data[0][2] * self.imu_data[0][2])
        yaw = math.atan2(t3,t4)*self.R2D
        
        # ROT 45 -> roll 
        rot_x = t1*math.cos(45.0*np.pi/180.0)
        #rot_y = t1*math.sin(45.0*np.pi/180.0)
        #rot_z = t0
        rot45_roll = math.atan2(t0,rot_x)*self.R2D

        #update IMU roll, pitch, yaw
        self.imu_data[3][0] = roll
        self.imu_data[3][1] = pitch
        self.imu_data[3][2] = yaw 
        self.imu_data[3][3] = rot45_roll
        self.IMU_SINC()

    def callback(self, data):   # IMU callback
        self.imu_data[0][0] = data.orientation.x
        self.imu_data[0][1] = data.orientation.y
        self.imu_data[0][2] = data.orientation.z
        self.imu_data[0][3] = data.orientation.w
        self.imu_data[1][0] = data.angular_velocity.x
        self.imu_data[1][1] = data.angular_velocity.y
        self.imu_data[1][2] = data.angular_velocity.z
        self.imu_data[2][0] = data.linear_acceleration.x
        self.imu_data[2][1] = data.linear_acceleration.y
        self.imu_data[2][2] = data.linear_acceleration.z
        self.QuaterniontoEuler()

    def IMU_SINC(self): # Imu time sinc
        for k in range(1, self.imu_sinc):
            self.imu_save[k-1][0] = self.imu_save[k][0]
            self.imu_save[k-1][1] = self.imu_save[k][1]
            self.imu_save[k-1][2] = self.imu_save[k][2]
            self.imu_save[k-1][3] = self.imu_save[k][3]

        self.imu_save[self.imu_sinc-1][0] = self.imu_data[3][0]
        self.imu_save[self.imu_sinc-1][1] = self.imu_data[3][1]
        self.imu_save[self.imu_sinc-1][2] = self.imu_data[3][2]
        self.imu_save[self.imu_sinc-1][3] = self.imu_data[3][3]

    def Camera_init(self, camnum):  # camera init

        if camnum==0:           ## Camera0
            self.cap0 = cv2.VideoCapture(self.cam0_addr)
            self.cap0.set(3,self.ImgWidth)
            self.cap0.set(4,self.ImgHeight)
            self.cap0.set(5,self.ImgFps)
        elif camnum==1:         ## Camera1
            self.cap1 = cv2.VideoCapture(self.cam1_addr)
            self.cap1.set(3,self.ImgWidth)
            self.cap1.set(4,self.ImgHeight)
            self.cap1.set(5,self.ImgFps)
        # elif camnum==2:         ## Camera2
        #     self.cap2 = cv2.VideoCapture(self.cam2_addr)
        #     self.cap2.set(3,self.ImgWidth)
        #     self.cap2.set(4,self.ImgHeight)
        #     self.cap2.set(5,self.ImgFps)
        
        # Set exposure_time
#                     brightness 0x00980900 (int)    : min=-15 max=15 step=1 default=0 value=0
#                       contrast 0x00980901 (int)    : min=0 max=30 step=1 default=10 value=10
#                     saturation 0x00980902 (int)    : min=0 max=60 step=1 default=16 value=16
# white_balance_temperature_auto 0x0098090c (bool)   : default=0 value=1
#                          gamma 0x00980910 (int)    : min=40 max=500 step=1 default=220 value=220
#                           gain 0x00980913 (int)    : min=1 max=100 step=1 default=1 value=1
#      white_balance_temperature 0x0098091a (int)    : min=1000 max=10000 step=50 default=4500 value=4500 flags=inactive
#                      sharpness 0x0098091b (int)    : min=0 max=127 step=1 default=16 value=16
#                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=0 value=1
#              exposure_absolute 0x009a0902 (int)    : min=1 max=10000 step=1 default=312 value=1
#                   pan_absolute 0x009a0908 (int)    : min=-648000 max=648000 step=3600 default=0 value=0
#                  tilt_absolute 0x009a0909 (int)    : min=-648000 max=648000 step=3600 default=0 value=0
#                  zoom_absolute 0x009a090d (int)    : min=100 max=800 step=1 default=102 value=102

        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=exposure_auto=0".format(self.cam4_addr)],        shell=True)
        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=exposure_absolute=312".format(self.cam4_addr)],   shell=True)
        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=white_balance_temperature_auto=0".format(self.cam4_addr)],   shell=True)
        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=gamma=100".format(self.cam4_addr)],   shell=True)
        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=gain=50".format(self.cam4_addr)],   shell=True)
        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=gamma=40".format(self.cam4_addr)],   shell=True)
        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=gamma=40".format(self.cam4_addr)],   shell=True)
        # subprocess.check_call(["v4l2-ctl -d {} --set-ctrl=white_balance_temperature=4500".format(self.cam4_addr)],   shell=True)
          

        print(" CAM" + str(camnum) + " is initialized!! ")

    def Display(self, img, CAM_num):
        
        thickness = 2
        height, width = img.shape[:2]
        # print("height : " + str(height) + " width : " + str(width) )
        Ang_height = 25
        Deg_10_height = 0.5 * 10 / Ang_height

        if CAM_num==0 or CAM_num==3 :
            cos_roll = math.cos(-1*self.imu_save[0][0]*self.D2R)
            sin_roll = math.sin(-1*self.imu_save[0][0]*self.D2R)

            cos_pitch = math.cos(-1*self.imu_save[0][1]*self.D2R)
            sin_pitch = math.sin(-1*self.imu_save[0][1]*self.D2R)
            alt_exam = 100

            # horizontal line
            img = cv2.line(img, (int(0.5*width - 0.3*width*cos_roll), int(0.5*height - 0.3*width*sin_roll)),  (int(0.5*width + 0.3*width*cos_roll), int(0.5*height + 0.3*width*sin_roll)), (0,255,0), thickness)

            # Center circle
            img = cv2.circle(img, (int(0.5*width), int(0.5*height)), int(0.01*width), (0,255,0), thickness)
            img = cv2.line(  img, (int(0.5*width - 0.03*width), int(0.5*height)), (int(0.5*width - 0.01*width), int(0.5*height)),   (0,255,0), 3) 
            img = cv2.line(  img, (int(0.5*width + 0.03*width), int(0.5*height)), (int(0.5*width + 0.01*width), int(0.5*height)),   (0,255,0), 3) 
            img = cv2.line(  img, (int(0.5*width),   int(0.5*height-0.01*width)), (int(0.5*width),   int(0.5*height-0.02*width)),   (0,255,0), 3) 

            # Altitude indicator
            img = cv2.putText(img, str(alt_exam), (int(0.01*width), int(0.52*height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)    
            img = cv2.line(  img, (int(0.02*width), int((0.5-0.3)*height)), (int(0.05*width), int((0.5-0.3)*height)), (0,255,0), 1) 
            img = cv2.line(  img, (int(0.02*width), int((0.5+0.3)*height)), (int(0.05*width), int((0.5+0.3)*height)), (0,255,0), 1)
            img = cv2.line(  img, (int(0.05*width), int((0.5-0.3)*height)), (int(0.05*width), int((0.5+0.3)*height)), (0,255,0), 1)

            # Vertical indicator
                # +10 degree
            img = cv2.putText(img, '+10', (int(0.5*width + (0.1*width*cos_roll + Deg_10_height*height*sin_roll)), int(0.5*height + (0.1*width*sin_roll - Deg_10_height*height*cos_roll))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)    
            img = cv2.line(img, (int(0.5*width + (-0.1*width*cos_roll + Deg_10_height*height*sin_roll)), int(0.5*height + (-0.1*width*sin_roll - Deg_10_height*height*cos_roll))), (int(0.5*width + (0.1*width*cos_roll + Deg_10_height*height*sin_roll)), int(0.5*height + (0.1*width*sin_roll - Deg_10_height*height*cos_roll))),      (0,255,0), thickness)
                # -10 degree
            img = cv2.putText(img, '-10', (int(0.5*width + (0.1*width*cos_roll - Deg_10_height*height*sin_roll)), int(0.5*height + (0.1*width*sin_roll + Deg_10_height*height*cos_roll))), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2) 
            img = cv2.line(img, (int(0.5*width + (-0.1*width*cos_roll - Deg_10_height*height*sin_roll)), int(0.5*height + (-0.1*width*sin_roll + Deg_10_height*height*cos_roll))), (int(0.5*width + (0.1*width*cos_roll - Deg_10_height*height*sin_roll)), int(0.5*height + (0.1*width*sin_roll + Deg_10_height*height*cos_roll))),      (0,255,0), thickness)

            # Roll indicator
            roll_indi_width  = 0.5*width
            roll_indi_height = 0.2*height
            point_30_width   = 0.1*width*math.cos(60*self.D2R)
            point_30_height  = 0.1*width*math.sin(60*self.D2R)
            point_60_width   = 0.1*width*math.cos(30*self.D2R)
            point_60_height  = 0.1*width*math.sin(30*self.D2R)
            thickness_roll   = 5

            # Degree 0
            img = cv2.line(img, (int(roll_indi_width), int(roll_indi_height-0.1*width)), (int(roll_indi_width), int(roll_indi_height-0.1*width*1.05)), (0,255,0), thickness_roll)
            img = cv2.putText(img, str( 0), (int(roll_indi_width), int(roll_indi_height-0.1*width)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)    

            # Degree +-30
            img = cv2.line(img, (int(roll_indi_width+point_30_width), int(roll_indi_height-point_30_height)), (int(roll_indi_width+point_30_width*1.05), int(roll_indi_height-point_30_height*1.05)), (0,255,0), thickness_roll)
            img = cv2.line(img, (int(roll_indi_width-point_30_width), int(roll_indi_height-point_30_height)), (int(roll_indi_width-point_30_width*1.05), int(roll_indi_height-point_30_height*1.05)), (0,255,0), thickness_roll)
            img = cv2.putText(img, str(  -30), (int(roll_indi_width+point_30_width), int(roll_indi_height-point_30_height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)    
            img = cv2.putText(img, str(  +30), (int(roll_indi_width-point_30_width), int(roll_indi_height-point_30_height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

            # Degree +-60
            img = cv2.line(img, (int(roll_indi_width+point_60_width), int(roll_indi_height-point_60_height)), (int(roll_indi_width+point_60_width*1.05), int(roll_indi_height-point_60_height*1.05)), (0,255,0), thickness_roll)
            img = cv2.line(img, (int(roll_indi_width-point_60_width), int(roll_indi_height-point_60_height)), (int(roll_indi_width-point_60_width*1.05), int(roll_indi_height-point_60_height*1.05)), (0,255,0), thickness_roll)
            img = cv2.putText(img, str(  -60), (int(roll_indi_width+point_60_width), int(roll_indi_height-point_60_height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)    
            img = cv2.putText(img, str(  +60), (int(roll_indi_width-point_60_width), int(roll_indi_height-point_60_height)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

            # Arrow   self.imu_save[0][0]*self.D2R
            img = cv2.line(img, (int(roll_indi_width+0.07*width*math.sin(-1*self.imu_save[0][0]*self.D2R)), int(roll_indi_height-0.07*width*math.cos(-1*self.imu_save[0][0]*self.D2R))), (int(roll_indi_width+0.05*width*math.sin(-1*self.imu_save[0][0]*self.D2R)), int(roll_indi_height-0.05*width*math.cos(-1*self.imu_save[0][0]*self.D2R))), (0,255,0), thickness)

        elif CAM_num==1 or CAM_num==4:
            img = cv2.line(img, (int(0.5*width - 0.3*width*math.cos(-1*self.imu_save[0][3]*self.D2R)), int(0.5*height - 0.3*width*math.sin(-1*self.imu_save[0][3]*self.D2R))),  (int(0.5*width + 0.3*width*math.cos(-1*self.imu_save[0][3]*self.D2R)), int(0.5*height + 0.3*width*math.sin(-1*self.imu_save[0][3]*self.D2R))), (0,255,0), thickness)
        
        elif CAM_num==2 or CAM_num==5:
            img = cv2.line(img, (int(0.5*width - 0.3*width*math.cos(self.imu_save[0][1]*self.D2R)), int(0.5*height - 0.3*width*math.sin(self.imu_save[0][1]*self.D2R))),  (int(0.5*width + 0.3*width*math.cos(self.imu_save[0][1]*self.D2R)), int(0.5*height + 0.3*width*math.sin(self.imu_save[0][1]*self.D2R))), (0,255,0), thickness)

        return img


    def run(self):
        # Initialize Camera 
        # Left cam
        self.Camera_init( 0 )
        self.Camera_init( 1 )
        # self.Camera_init( 2 )
       
        # Make imgsave folder
        if self.save_img == 1:
            if ASUS:
                saving_folder   = "/home/usrg/Civil/img_test_1/"
            if STICK:
                saving_folder   = "/media/usrg/New\ Volume/"
            local_folder    = time.localtime()
            local_folder    = "%04d-%02d-%02d-%02d:%02d:%02d" % (local_folder.tm_year, local_folder.tm_mon, local_folder.tm_mday, local_folder.tm_hour, local_folder.tm_min, local_folder.tm_sec)
            path_image0     = saving_folder+local_folder+'/frame0/'
            path_image1     = saving_folder+local_folder+'/frame1/'
            # path_image2     = saving_folder+local_folder+'/frame2/'
            
            if not(os.path.isdir('./'+local_folder)):
                os.makedirs(os.path.join(saving_folder+local_folder))
                os.makedirs(os.path.join(path_image0))
                os.makedirs(os.path.join(path_image1))
                # os.makedirs(os.path.join(path_image2))
             
        # Ros init
        rospy.init_node('image_gmsl', anonymous=True)

        # Define Publisher & Subscriber
        # image_pub1 = rospy.Publisher("image_frame4",        Image,  queue_size=1)
        image_pub1 = rospy.Publisher("image_frame1",        Image,  queue_size=1)
        image_pub2 = rospy.Publisher("image_frame2",        Image,  queue_size=1)
        
        imu_sub    = rospy.Subscriber("/mavros/imu/data",   Imu,    self.callback)
        bridge = CvBridge()

        # Image read
        ret0, frame0 = self.cap0.read()
        ret1, frame1 = self.cap1.read()
        # ret2, frame2 = self.cap2.read()

        while not rospy.is_shutdown():
            # Image read
            now = time.time()
            print("ROS TIME :" + str(now) )
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()
            # ret2, frame2 = self.cap2.read()

            if ret0 is True or ret1 is True:
                # now = time.time()

                # Publish Image    
                try:
                    if ret0 is True: 
                        image_pub1.publish(bridge.cv2_to_imgmsg(frame0, "bgr8"))  # "bgr8"  "mono8"
                    if ret1 is True: 
                        image_pub2.publish(bridge.cv2_to_imgmsg(frame1, "bgr8"))  # "bgr8"  "mono8"
                    # if ret2 is True: 
                    #     image_pub3.publish(bridge.cv2_to_imgmsg(frame2, "bgr8"))  # "bgr8"  "mono8"
                except CvBridgeError as e:
                    print(e)
                except KeyboardInterrupt:
                    print("Shutting down")
                    break

                # Save Image
                if self.save_img == 1:
                    if ret0 is True: 
                        t0 = threading.Thread(target=self.saveImg, args=(path_image0+str(now)+'.jpg', frame0))
                        t0.start()
                    if ret1 is True: 
                        t1 = threading.Thread(target=self.saveImg, args=(path_image1+str(now)+'.jpg', frame1))
                        t1.start()
                    # if ret2 is True: 
                    #     t2 = threading.Thread(target=self.saveImg, args=(path_image2+str(now)+'.jpg', frame2))
                    #     t2.start()
                    

                # Image processing  #################################### This is test using one camera. plz, back to original
                # if ret0 is True:    
                #     # frame0 = self.img_processing(frame0, 0)
                #     frame0 = self.Display(frame0, 0)        
                # if ret1 is True:
                #     # frame1 = self.img_processing(frame1, 1)
                #     frame1 = self.Display(frame1, 1)  
                # if ret2 is True:
                #     # frame2 = self.img_processing(frame2, 2)
                #     frame2 = self.Display(frame2, 2)  
                # if ret3 is True:    
                #     # frame3 = self.img_processing(frame3, 3)
                #     frame3 = self.Display(frame3, 3)        
                # if ret4 is True:
                #     # frame4 = self.img_processing(frame4, 4)
                #     frame4 = self.Display(frame4, 4)  
                # if ret5 is True:
                #     # frame5 = self.img_processing(frame5, 5)
                #     frame5 = self.Display(frame5, 5)  


                # Calculate fps
                self.fps = 1.0/(now-self.prev_time)
                print(str(self.fps) + " fps")
                
                # Update previous time for calculating fps
                self.prev_time = now

            else:
                continue

        self.cap0.release()
        self.cap1.release()
        # self.cap2.release()

        cv2.destroyAllWindows()
        


################ MAIN ###################

Det = Detection()
Det.run()


###Nail Camera Property
# pixfmt 0 = 'UYVY' desc = 'UYVY 4:2:2'
#   discrete: 1280x720:    1/60 1/30 
#   discrete: 1920x1080:   1/45 1/30 1/15 
#   discrete: 2304x1296:   1/30 1/15 
#   discrete: 2304x1536:   1/24 1/12 
#   discrete: 640x480:     1/60 1/30 
#   discrete: 1920x1280:   1/30 1/15 
#   discrete: 1152x768:    1/60 1/30 
#   discrete: 2048x1536:   1/29 1/15 
#   discrete: 1280x960:    1/58 1/30 

#                     brightness 0x00980900 (int)    : min=-15 max=15 step=1 default=0 value=0
#                       contrast 0x00980901 (int)    : min=0 max=30 step=1 default=10 value=10
#                     saturation 0x00980902 (int)    : min=0 max=60 step=1 default=16 value=16
# white_balance_temperature_auto 0x0098090c (bool)   : default=0 value=1
#                          gamma 0x00980910 (int)    : min=40 max=500 step=1 default=220 value=220
#                           gain 0x00980913 (int)    : min=1 max=100 step=1 default=1 value=1
#      white_balance_temperature 0x0098091a (int)    : min=1000 max=10000 step=50 default=4500 value=4500 flags=inactive
#                      sharpness 0x0098091b (int)    : min=0 max=127 step=1 default=16 value=16
#                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=0 value=1
#              exposure_absolute 0x009a0902 (int)    : min=1 max=10000 step=1 default=312 value=1
#                   pan_absolute 0x009a0908 (int)    : min=-648000 max=648000 step=3600 default=0 value=0
#                  tilt_absolute 0x009a0909 (int)    : min=-648000 max=648000 step=3600 default=0 value=0
#                  zoom_absolute 0x009a090d (int)    : min=100 max=800 step=1 default=102 value=102


###Video Capture Property
# 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
# 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
# 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
# 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
# 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
# 5. CV_CAP_PROP_FPS Frame rate.
# 6. CV_CAP_PROP_FOURCC 4-character code of codec.
# 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
# 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
# 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
# 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
# 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
# 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
# 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
# 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
# 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
# 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
# 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
# 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
