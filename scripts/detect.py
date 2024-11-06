#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
from numpy import unique
from numpy import where
import time
import math
from skimage import morphology
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import csv
import ctrlFunc as cf
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from std_msgs.msg import Float64MultiArray, Header

t = 0
arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
rospy.init_node('detect', anonymous=True)

def drawRefContours(t, frameUndistorted):
    s = np.arange(1001).reshape(1001, 1) * 0.001
    
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


def detectTube(frameUndistorted, a):
    x, y, w, h = 75, 50, 1106, 610
    roi_frame = frameUndistorted[y:y+h, x:x+w]
    
    roi_frame = cv2.GaussianBlur(roi_frame, (5, 5), 0)
        
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
    '''
    # Define the range for blue color in HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
        
    # Create a mask for blue color
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    '''
        
    # Define the range for red color in HSV
    lower_red1 = np.array([0, 80, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
    # Combine the masks
    mask = mask1 + mask2

    #cv2.imshow('mask', mask)

    # Detect the middle line of the tube
    ret, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8) * 255
    #cv2.imshow('skeleton', skeleton)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Draw the contours on the original frame and extract the point cloud of the tube
    point_cloud = np.empty((0, 2), int)
    print("length of contours: ", len(contours))
    for contour in contours:
        contour += [x, y]  # Adjust contour coordinates to match the original image
        print("contour length: ", len(contour))
        cv2.drawContours(frameUndistorted, [contour], -1, (0, 255, 0), 3)
        for point in contour:
            cv2.circle(frameUndistorted, point[0], 2, (255, 0, 0), -1)
            point_cloud = np.append(point_cloud, [point[0]], axis=0)
            
    print(len(point_cloud))
    
    # define the model
    model = MiniBatchKMeans(n_clusters=len(point_cloud)//2)

    # fit the model
    model.fit(point_cloud)

    # assign a cluster to each example
    yhat = model.predict(point_cloud)

    # retrieve unique clusters
    clusters = unique(yhat)

    # create scatter plot for samples from each cluster
    new_pc = np.zeros((len(clusters), 2))
    pc_out = [0] * 2 * (len(clusters))
    i = 0
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        new_pc[i, 0] = int(np.mean(point_cloud[row_ix, 0]))
        new_pc[i, 1] = int(np.mean(point_cloud[row_ix, 1]))
        pc_out[2*i] = new_pc[i, 0]
        pc_out[2*i+1] = new_pc[i, 1]
        i += 1
        
    ''' 
    fp = open('data.csv', mode='a+', newline='')
    dp = csv.writer(fp)
    dp.writerow(pc_out)
    fp.close()
    
    plt.scatter(new_pc[:, 0], new_pc[:, 1])
    #fig = plt.show()
    plt.pause(0.005)
    plt.ioff()
    plt.cla()
    #plt.close()
    '''
    
    for i in range(len(a)//4):
        new_pc = np.append(new_pc, [[a[4*i]/0.00228571, a[4*i+1]/0.00228571]], axis=0)
        
    new_pc = sortPoints(new_pc)
        
    gs = cf.generateDetectedGs(new_pc, 8)
    coff = cf.curveFitting(gs, new_pc.reshape(2*len(new_pc), 1), 8)
        
    gs_new = cf.generateGs(0, 0.001, 1, 0, 0, 8)
    pt_fitted = gs_new * coff
    points = np.array([pt_fitted[0::2], pt_fitted[1::2]]).T
    cv2.polylines(frameUndistorted, np.int32([points]), isClosed=False, color=(0, 0, 255), thickness=3)


def sortPoints(points):
    point_cloud = np.empty((0, 2), int)
    point = points[-1, :]
    
    while len(points) > 0:
        # Calculate the Euclidean distance from the point to each point in the vector
        distances = np.linalg.norm(points - point, axis=1)
    
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
    
        # Get the point with the minimum distance
        point_cloud = np.append(point_cloud, [points[min_index]], axis=0)
        points = np.delete(points, min_index, axis=0)
        point = point_cloud[-1, :]
    
    return point_cloud

def getPose(corners, ids):
    
    a = [0] * len(ids) * 4
    if ids is not None:
        for i in range(len(ids)):
            center_x = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4 * 0.00228571
            center_y = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4 * 0.00228571
            yaw = math.atan2(corners[i][0][0][1] - corners[i][0][3][1], corners[i][0][0][0] - corners[i][0][3][0])
            
            a[i*4] = center_x
            a[i*4+1] = center_y
            a[i*4+2] = yaw
            a[i*4+3] = ids[i][0]
    
    return a


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
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('video.avi', fourcc, 30.0, (1280, 720), True)
    
    image = cv2.imread('/home/binzhang/zbin13/codingsomethingcool/epuck_ws/src/epuck_driver_cpp_wifi/detect1.jpg')

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
    t1 = time.time()
    while not rospy.is_shutdown():
        #ret, frame = videoRgb.read()
        frame = cv2.resize(image, (1280, 720))
        cv2.imwrite('frame.jpg', frame)
        #if not ret:
        #    break
        frameUndistorted = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
        cv2.imshow('frame', frameUndistorted)
        cv2.waitKey(1)
        
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=parameters)
        
        a = getPose(corners, ids)
        
        # Detect the tube
        detectTube(frameUndistorted, a)
        
        frameMarkers = aruco.drawDetectedMarkers(frameUndistorted, corners, ids)
        
        # Draw the reference contours
        #drawRefContours(t, frameUndistorted)
        
        if ids is not None:
            
            print("length of ids: ", len(ids))
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.036, cameraMatrix, distCoeffs)
            if ids is None:
                continue
            else:
                for i in range(len(ids)):
                    rvec, tvec = rvecs[i], tvecs[i]
                    rvec = rvec.reshape((3, 1))
                    tvec = tvec.reshape((3, 1))
                    cv2.drawFrameAxes(frameUndistorted, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
                    cv2.imshow('frameMarkers', frameUndistorted)
                
                    pose = Pose()
                    pose.position.x = a[4*i]
                    pose.position.y = a[4*i+1]  
                    pose.position.z = 0
                    pose.orientation.x = 0
                    pose.orientation.y = 0
                    pose.orientation.z = a[4*i+2]
                    pose.orientation.w = ids[i][0]
                    poseArray = PoseArray()
                    poseArray.header = Header()
                    poseArray.header.stamp = rospy.Time.now()
                    poseArray.header.frame_id = 'aruco'
                    poseArray.poses.append(pose)
                    print("pose of id", ids[i][0], ":\n", pose)
                    bot_pub.publish(poseArray)
                    
        video.write(frameUndistorted)
        
        t2 = time.time()
        t = t2 - t1
        t1 = t2
        print("t: ", t)
        # Publish the detected data
        data = Float64MultiArray(data=a)
        pose_pub.publish(data)        
        rate.sleep()
