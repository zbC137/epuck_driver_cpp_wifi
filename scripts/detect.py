#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from std_msgs.msg import Float64MultiArray, Header

t = 0
arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
rospy.init_node('detect', anonymous=True)


def callback(data):
    global t
    t = data.data[0]

if __name__ == '__main__':
    bot_pub = rospy.Publisher('/robot', PoseArray, queue_size=10)
    pose_pub = rospy.Publisher('robot_pose', Float64MultiArray, queue_size=10)
    t_sub = rospy.Subscriber('time', Float64MultiArray, callback)
    rate = rospy.Rate(100)
    videoRgb = cv2.VideoCapture('/dev/video0')
    videoRgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    videoRgb.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    videoRgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Camera
    fx_r = 582.59118861
    fy_r = 582.65884802
    cx_r= 629.53535406
    cy_r = 348.71988126
    k1_r = 0.00239457
    k2_r = -0.03004914
    p1_r = -0.00062043
    p2_r = -0.00057221
    k3_r= 0.01083464
    
    cameraMatrix = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]], dtype=np.float32)
    distCoeffs = np.array([k1_r, k2_r, p1_r, p2_r, k3_r], dtype=np.float32)
    
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (1280, 720), 0, (1280, 720))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (1280, 720), cv2.CV_16SC2)
    
    s = np.arange(1001).reshape(1001, 1) * 0.001
    
    while not rospy.is_shutdown():
        ret, frame = videoRgb.read()
        #cv2.imwrite('frame.jpg', frame)
        if not ret:
            break
        frameUndistorted = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
        
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frameUndistorted, cv2.COLOR_BGR2HSV)
        '''
        # Define the range for blue color in HSV
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        
        # Create a mask for blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        '''
        
        # Define the range for green color in HSV
        lower_green = np.array([35, 50, 100])
        upper_green = np.array([85, 255, 255])
    
        # Create a mask for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        '''
        # Define the range for yellow color in HSV
        lower_yellow = np.array([20, 50, 100])
        upper_yellow = np.array([30, 255, 255])
    
        # Create a mask for yellow color
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        '''
        '''
        # Define the range for red color in HSV
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine the masks
        mask = mask1 + mask2
        '''
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the contours on the original frame
        cv2.drawContours(frameUndistorted, contours, -1, (0, 255, 0), 3)
        
        # Extract the point cloud of the tube
        point_cloud = []
        for contour in contours:
            for point in contour:
                point_cloud.append(point[0])
        
        # Draw the point cloud on the image
        for point in point_cloud:
            cv2.circle(frameUndistorted, tuple(point), 2, (255, 0, 0), -1)
        
        # Save the image with the detected blue tube
        #cv2.imwrite('frame_with_blue_tube.jpg', frameUndistorted)
        
        cv2.imshow('frame', frameUndistorted)
        cv2.waitKey(1)
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=parameters)
        frameMarkers = aruco.drawDetectedMarkers(frameUndistorted, corners, ids)
        
        if t < 30:
            x = 150 * np.cos(2*math.pi*s) + 850
            y = 150 * np.sin(2*math.pi*s) + 350
        elif t<60:
            x = 175 * np.cos(2 * math.pi * s) + 650
            y = 125 * np.sin(2 * math.pi * s) + 350
        else:
            x = 100 + (125 * (1 - np.sin(2 * math.pi * s))) * np.cos(2 * math.pi * s) + 350
            y = 100 + (125 * (1 - np.sin(2 * math.pi * s))) * np.sin(2 * math.pi * s) + 350
            
        points = np.array([x, y]).T
        cv2.polylines(frameUndistorted, np.int32([points]), isClosed=False, color=(0, 0, 255), thickness=3)
        
        a = [0] * len(ids) * 4
        if ids is not None or a is not None:
            
            print("length of ids: ", len(ids))
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.036, cameraMatrix, distCoeffs)
            if ids is None:
                continue
            else:
                for i in range(len(ids)):
                    rvec, tvec = rvecs[i], tvecs[i]
                    rvec = rvec.reshape((3, 1))
                    tvec = tvec.reshape((3, 1))
                    rotationMatrix, _ = cv2.Rodrigues(rvec)
                    quaternion = tft.quaternion_from_matrix(np.vstack((np.hstack((rotationMatrix, [[0], [0], [0]])), [0, 0, 0, 1])))
                    euler = tft.euler_from_quaternion(quaternion)
                    roll, pitch, yaw = euler[0], euler[1], euler[2]
                    cv2.drawFrameAxes(frameUndistorted, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
                    cv2.imshow('frameMarkers', frameMarkers)
                
                    center_x = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4 * 0.00228571
                    center_y = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4 * 0.00228571
                    yaw = math.atan2(corners[i][0][0][1] - corners[i][0][3][1], corners[i][0][0][0] - corners[i][0][3][0])
                
                    pose = Pose()
                    pose.position.x = center_x
                    pose.position.y = center_y
                    pose.position.z = 0
                    pose.orientation.x = 0
                    pose.orientation.y = 0
                    pose.orientation.z = yaw
                    pose.orientation.w = ids[i][0]
                    poseArray = PoseArray()
                    poseArray.header = Header()
                    poseArray.header.stamp = rospy.Time.now()
                    poseArray.header.frame_id = 'aruco'
                    poseArray.poses.append(pose)
                    print("pose of id", ids[i][0], ":\n", pose)
                    bot_pub.publish(poseArray)

                    a[i*4] = center_x
                    a[i*4+1] = center_y
                    a[i*4+2] = yaw
                    a[i*4+3] = ids[i][0]

        data = Float64MultiArray(data=a)
        pose_pub.publish(data)        
        rate.sleep()
