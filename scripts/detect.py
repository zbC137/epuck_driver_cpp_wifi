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

arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
rospy.init_node('detect', anonymous=True)


if __name__ == '__main__':
    bot_pub = rospy.Publisher('/robot', PoseArray, queue_size=10)
    pose_pub = rospy.Publisher('robot_pose', Float64MultiArray, queue_size=10)
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
    
    while not rospy.is_shutdown():
        ret, frame = videoRgb.read()
        #cv2.imwrite('frame.jpg', frame)
        if not ret:
            break
        frameUndistorted = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
        cv2.imshow('frame', frameUndistorted)
        cv2.waitKey(1)
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=parameters)
        frameMarkers = aruco.drawDetectedMarkers(frameUndistorted, corners, ids)
        
        a = [0] * 2 * 4
        if ids is not None or a is not None:
            
            #print("length of ids: ", len(ids))
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
                
                    center_x = (corners[0][i][0][0] + corners[0][i][1][0] + corners[0][i][2][0] + corners[0][i][3][0]) / 4 * 0.00228571
                    center_y = (corners[0][i][0][1] + corners[0][i][1][1] + corners[0][i][2][1] + corners[0][i][3][1]) / 4 * 0.00228571
                    yaw = math.atan2(corners[0][i][0][1] - corners[0][i][3][1], corners[0][i][0][0] - corners[0][i][3][0])
                
                    pose = Pose()
                    '''
                    pose.position.x = tvec[0][0]
                    pose.position.y = tvec[1][0]
                    pose.position.z = tvec[2][0]
                    pose.orientation.x = roll
                    pose.orientation.y = pitch
                    pose.orientation.z = yaw
                    pose.orientation.w = ids[i][0]
                    '''
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
