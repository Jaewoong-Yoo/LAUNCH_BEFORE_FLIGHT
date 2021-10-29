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

import imutils
import roslib

import sys
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

#For ROS Subscribe
from sensor_msgs.msg import Imu

file_number = 1
desktop_name = 'usrg'

class Detection:
    def __init__(self):
        #Define Variable###################################################################################
        self.save_img        = 0        # 0 : Don't save, 1: Save
        self.imu_sinc        = 13       # IMU time sinc

        self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_32490400-video-index0"  # right 0 / 5 (board pose)
        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_1A480200-video-index0"  # left 45 / 4 (board pose)
        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_1A080800-video-index0"  # right 90 / 6 (board pose)

        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_114F0B00-video-index0"  # left 90 / 2 (board pose)
        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_14480200-video-index0"  # left 0 / 1 (board pose)
        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_2B460000-video-index0"  # right 45 / 3 (board pose)
        # self.cam0_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_23490400-video-index0" #left
        # self.cam1_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_28070800-video-index0" #front
        # self.cam2_addr = "/dev/v4l/by-id/usb-e-con_systems_NileCAM30_USB_24490400-video-index0" #right
        ###################################################################################################

        #Define list
        self.ImgWidth        = 1920 # Resolution 
        self.ImgHeight       = 1080 # Resolution
        self.ImgFps          = 15   # Should be modified with fps
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
        self.frame_idx0      = 0

        # Create some random colors
        self.color           = np.random.randint(0,255,(100,3))
        self.OF_CAM0_count   = 0
        self.Prev_CMO_img0 = []

        # Variables for LOG
        # self.
    def Camera_init(self, camnum):  # camera init

        if camnum==0:           ## Camera0
            self.cap0 = cv2.VideoCapture(self.cam0_addr)
            self.cap0.set(3,self.ImgWidth)
            self.cap0.set(4,self.ImgHeight)
            self.cap0.set(5,self.ImgFps)
        print(" CAM" + str(camnum) + " is initialized!! ")

    def saveImg(self, name, img):
        cv2.imwrite(name, img)

    def run(self):
        # Initialize Camera 
        # Left cam
        self.Camera_init( 0 )
        
        # Make folder
        if self.save_img == 1:
            saving_folder   = "/home/" + desktop_name + "/Civil/img_test_" + str(file_number) + "/"
            local_folder    = time.localtime()
            local_folder    = "%04d-%02d-%02d-%02d:%02d:%02d" % (local_folder.tm_year, local_folder.tm_mon, local_folder.tm_mday, local_folder.tm_hour, local_folder.tm_min, local_folder.tm_sec)
            path_image0     = saving_folder+local_folder+'/frame0/'

            if not(os.path.isdir('./'+local_folder)):
                os.makedirs(os.path.join(saving_folder+local_folder))
                os.makedirs(os.path.join(path_image0))
        
        # Ros init
        rospy.init_node('image_gmsl', anonymous=True)

        # Define Publisher & Subscriber
        # image_pub1 = rospy.Publisher("image_frame" + str(file_number),        Image,  queue_size=1)
        image_pub1 = rospy.Publisher("image_frame" + str(file_number) + "/compressed",          CompressedImage, queue_size=1)
        bridge = CvBridge()

        # Image read
        ret0, frame0 = self.cap0.read()

        while not rospy.is_shutdown():
            # Image read
            now = time.time()
            ret0, frame0 = self.cap0.read()

            if ret0 is True:
                # now = time.time()
                # Publish Image    
                try:
                    if ret0 is True: 
                        # raw_image = bridge.cv2_to_imgmsg(frame0, "bgr8")
                        compressed_image = bridge.cv2_to_compressed_imgmsg(frame0)
                        image_pub1.publish(compressed_image)

                        # image_pub1.publish(bridge.cv2_to_imgmsg(frame0, "bgr8"))  # "bgr8"  "mono8"
                except CvBridgeError as e:
                    print(e)
                except KeyboardInterrupt:
                    print("Shutting down")
                    break

                if self.save_img == 1:
                    if ret0 is True: 
                        t0 = threading.Thread(target=self.saveImg, args=(path_image0+str(now)+'.jpg', frame0))
                        t0.start()

                # Calculate fps
                self.fps = 1.0/(now-self.prev_time)
                print(str(self.fps) + " fps")
                
                # Update previous time for calculating fps
                self.prev_time = now


                # Image processing  #################################### This is test using one camera. plz, back to original
                # if ret0 is True:    
                    # frame0 = self.img_processing(frame0, 0)
                    # frame0 = self.Display(frame0, 0)        
            else:
                continue

        self.cap0.release()
        cv2.destroyAllWindows()
        


################ MAIN ###################

Det = Detection()
Det.run()


