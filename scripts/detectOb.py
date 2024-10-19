#!/usr/bin/env python3
import cv2
import numpy as np
import time
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
#from sklearn.cluster import KMeans
import cv2.aruco as aruco
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseArray
import math

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
roicircle =[]
maskcircle = []
triangle_data = []
rectangle_data = []
corners = []
ARUIDS = []
rospy.init_node('OBpose''arIDS')
time1 = time.time()

circle_number=1
triangle_number=1
rectangle_number=4
# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280,720), True)
out_therf = cv2.VideoWriter('output_therf.avi', fourcc, 30.0, (640,512), True)
out_rgb = cv2.VideoWriter('output_rgb.avi', fourcc, 30.0, (1280,720), True)
shapeall = []
index = 0
numOB = 6 # the number of obstacles

def is_close(contour1, contour2, threshold):
    # �����������������ĵ�
    M1 = cv2.moments(contour1)
    M2 = cv2.moments(contour2)
    if M1["m00"] != 0 and M2["m00"] != 0:
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        cx2 = int(M2["m10"] / M2["m00"])
        cy2 = int(M2["m01"] / M2["m00"])
        distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        return distance < threshold
    return False

def recognize_shapes(image_path):
    #image = cv2.imread(image_path)
    image = image_path
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    blurred = cv2.GaussianBlur(gray, (1, 1), 5)
    edged = cv2.Canny(blurred, 30, 150) 
    cv2.imshow("edged",edged)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    lim=6000
    boolindex = 0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(contour)

        if len(approx) == 3 or len(approx) == 4:  
            if area > lim:
                if not any(is_close(contour, other['contour'], 50) for other in shapes):
                    shape = {
                        'contour': contour,
                        'approx': approx,
                        'type': 'Triangle' if len(approx) == 3 else 'Rectangle'
                    }
                    shapes.append(shape)

        else:  # 
            
            if area > lim:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 0.5 < circularity < 1.4:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    if not any(is_close(contour, other['contour'], 50) for other in shapes):
                        shape = {
                            'contour': contour,
                            'center': center,
                            'radius': radius,
                            'type': 'Circle'
                        }
                        shapes.append(shape)
    
    shapeall = shapes  #new code
    shapes = shapeall
    mask = np.zeros(image.shape[:2], dtype = np.uint8)
    tempimg = np.copy(imgnew)
    
    for shape in shapes:
        if len(shapes) == numOB:
            boolindex = 1
            area = cv2.contourArea(shape['contour'])
            if area > lim:
                if shape['type'] == 'Circle':
                    cv2.circle(image, shape['center'], shape['radius'], (255, 0, 0), 2)
                    cv2.circle(image, shape['center'], 5, (255, 0, 0), -1)
                    cv2.circle(mask,shape['center'],shape['radius'],255,-1)
                    print(f"Circle center: {shape['center']}, radius: {shape['radius']}")
                else:
                    cv2.drawContours(image, [shape['contour']], 0, (0, 255, 0) if shape['type'] == 'Triangle' else (0, 0, 255), 2)
                    for vert in shape['approx']:
                        x, y = vert.ravel()
                        cv2.circle(image, (x, y), 5, (0, 255, 0) if shape['type'] == 'Triangle' else (0, 0, 255), -1)
                        cv2.fillConvexPoly(mask, shape['approx'], 255)
                    print(f"{shape['type']} vertices: {shape['approx'].ravel()}")
                    
                    if shape['type'] == 'Triangle':
                        triangle_data.append(shape['approx'].ravel())
                        print("triangle_data",triangle_data)
                    elif shape['type'] == 'Rectangle':
                        rectangle_data.append(shape['approx'].ravel())
                        print("rectangle_data",rectangle_data[0][1])

    shapeall = shapes
    print("len(shapeall)",len(shapeall))                                        
    mask = cv2.GaussianBlur(mask, (15, 15), 39)
    #cv2.imshow('ddddddddd',mask)
    tempimg = cv2.bitwise_and(tempimg,tempimg,mask=mask)
    
    gray_img = cv2.cvtColor(tempimg,cv2.COLOR_RGB2GRAY)
    _,mask = cv2.threshold(gray_img,limit,255,cv2.THRESH_BINARY)
    tempimg = cv2.bitwise_and(tempimg,tempimg,mask=mask)       
    #imgmerge = cv2.add(tempimg,img1n)   
 
    # detector = aruco.ArucoDetector(dictionary,parameters)
    # markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)

    #return imgmerge
    return mask, boolindex



def create_robot_pose_array(corners,markerIds):
    # Create a PoseArray message
    pose_array = PoseArray()
    pose_array.header = Header()
    pose_array.header.stamp = rospy.Time.now()
    pose_array.header.frame_id = "map"

    # Calculate the center of the ArUco marker corners
    center_x = (corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]) / 4
    center_y = (corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]) / 4

    # Create a Pose message and set the position to the center of the ArUco marker
    pose = Pose()
    pose.position.x = center_x
    pose.position.y = center_y
    pose.position.z = markerIds  # Assuming the marker is on a flat surface

    # Calculate the orientation (yaw) of the marker
    yaw = math.atan2(corners[0][0][0][1] - corners[0][0][3][1], corners[0][0][0][0] - corners[0][0][3][0])
    pose.orientation.z = yaw 
    pose.orientation.w = 0

    pose_array.poses.append(pose)

    return pose_array

    
def publisher(): 
    #pub = rospy.Publisher('OBpose', PoseStamped, queue_size=10) 
    pub = rospy.Publisher('OBpose', Float64MultiArray, queue_size=10) 
    rate = rospy.Rate(100) 
    #while not rospy.is_shutdown(): 
    data = PoseStamped()
    i = 0
    a = [0] * ((circle_number+1)*3+rectangle_number*8 + triangle_number*6)
    if circle_number > 0:
        for circle in roicircle:
            a[3*i] = circle[0]
            a[3*i+1] = circle[1]
            a[3*i+2] = circle[2]
            i = i + 1
    if triangle_number > 0:
        for i in range(triangle_number):
            for j in range(6):
                a[3*circle_number + 6*i + j] = triangle_data[i][j]
    if rectangle_number > 0:
        for i in range(rectangle_number):
            for j in range(8):
                a[3*circle_number + 6*triangle_number + 8*i + j] = rectangle_data[i][j] 
        #print(data)
    a[-3] =(corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4
    a[-2] =(corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4
    a[-1] = math.atan2(corners[0][0][0][1]-corners[0][0][3][1], corners[0][0][0][0]-corners[0][0][3][0])
    
    #a = [roicircle[0], [x, y, w]]
    data=Float64MultiArray(data=a)
    rospy.loginfo(data) 
    pub.publish(data) 
    time2 = time.time()
    #print("time: ",time2 - time1)
    rate.sleep() 


if __name__ == '__main__': 
    
    pub = rospy.Publisher('/robot', PoseArray, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz
    videorgb = cv2.VideoCapture('/dev/video2')#11.mp4
    videothermal = cv2.VideoCapture('/dev/video0')
    #videoabc = cv2.VideoCapture('/dev/video4')
    videorgb.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    videothermal.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #videoabc.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    videorgb.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    videorgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
    #videoabc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
    #videoabc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
    videothermal.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    videothermal.set(cv2.CAP_PROP_FRAME_WIDTH, 512) 
    fx_t = 3.3789900820893826e+02
    fy_t = 3.3789900820893826e+02
    cx_t = 3.1950000000000000e+02
    cy_t = 2.5550000000000000e+02
    k1_t = -2.0905036755393389e-01
    k2_t = 4.8456933948032764e-02
    p1_t = 0.0
    p2_t = 0.0
    k3_t = -5.3895705714326197e-03

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



    cameraMatrix_t = np.array([[fx_t,0.0,cx_t],[0.0,fy_t,cy_t],[0.0,0.0,1]],dtype=np.float32)
    distCoeffs_t = np.array([k1_t,k2_t,p1_t,p2_t,k3_t],dtype=np.float32)

    cameraMatrix_r = np.array([[fx_r,0.0,cx_r],[0.0,fy_r,cy_r],[0.0,0.0,1]],dtype=np.float32)
    distCoeffs_r = np.array([k1_r,k2_r,p1_r,p2_r,k3_r],dtype=np.float32)

    dz = (640,512)
    dz2 = (1280,720)
    dz3 = (640,360)

    newCameraMatrix_t, _= cv2.getOptimalNewCameraMatrix(cameraMatrix_t,distCoeffs_t,dz,0,dz)
    map1_t,map2_t = cv2.initUndistortRectifyMap(cameraMatrix_t,distCoeffs_t,None,newCameraMatrix_t,dz,cv2.CV_16SC2)

    newCameraMatrix_r, _= cv2.getOptimalNewCameraMatrix(cameraMatrix_r,distCoeffs_r,dz2,0,dz2)
    map1_r,map2_r = cv2.initUndistortRectifyMap(cameraMatrix_r,distCoeffs_r,None,newCameraMatrix_r,dz2,cv2.CV_16SC2)
 
    h = np.array([[1.041242404901222, -0.04238103244525442, -15.76970287260019],
                    [0.01303887712116861, 0.9363200889631627, -0.4837570360693626],
                    [-2.241489635524001e-05, -0.0001023352413176867, 1]])
    dz = (640,512)
    dz2 = (1280,720)
    dz3 = (640,360)

    limit =20
    x = 0
    y = 80
    xw = 640
    yw = 360
    frame_index = 0
    kw = 0
    roipoint = []
    TempID = None
    TempCor = None
    TempRE = None
    maskfinal = np.zeros((720,1280), dtype = np.uint8)
    #if not videorgb.isOpened():
    #    print("ERRORrgb")
    if not videothermal.isOpened():
        print("ERRORthermal")






    # # Camera
    # fx_r = 582.59118861
    # fy_r = 582.65884802
    # cx_r= 629.53535406
    # cy_r = 348.71988126
    # k1_r = 0.00239457
    # k2_r = -0.03004914
    # p1_r = -0.00062043
    # p2_r = -0.00057221
    # k3_r= 0.01083464


    # cameraMatrix_r = np.array([[fx_r,0.0,cx_r],[0.0,fy_r,cy_r],[0.0,0.0,1]],dtype=np.float32)
    # distCoeffs_r = np.array([k1_r,k2_r,p1_r,p2_r,k3_r],dtype=np.float32)

    # dz = (640,512)
    # dz2 = (1280,720)
    # dz3 = (640,360)

    # newCameraMatrix_r, _= cv2.getOptimalNewCameraMatrix(cameraMatrix_r,distCoeffs_r,dz2,0,dz2)
    # map1_r,map2_r = cv2.initUndistortRectifyMap(cameraMatrix_r,distCoeffs_r,None,newCameraMatrix_r,dz2,cv2.CV_16SC2)
 

    while True:

        ret1,imgrgb = videorgb.read()
        ret2,imgthermal = videothermal.read()
        img1n = cv2.remap(imgrgb,map1_r,map2_r,interpolation=cv2.INTER_LINEAR)
        img2n = cv2.remap(imgthermal,map1_t,map2_t,interpolation=cv2.INTER_LINEAR)
        imgthermal = cv2.undistort(imgthermal, cameraMatrix_t, distCoeffs_t , None, newCameraMatrix_t)
        #cv2.imshow("therf",imgthermal)
        imgrgb = cv2.undistort(imgrgb, cameraMatrix_r, distCoeffs_r , None, newCameraMatrix_r)
        #cv2.imshow("rgb",img1n)
        out_therf.write(imgthermal)
        out_rgb.write(img1n)

        # cv2.imshow("rgb",img1n)
        # cv2.imshow("ther",img2n)
        # cv2.imshow("rgbf",imgrgb)
        # cv2.imshow("therf",imgthermal)
        x = 0
        y = 80
        xw = 640
        yw = 360
        img2new = imgthermal[y:y+yw,x:x+xw]
        #print("y:y+yw,x:x+xw",y,y+yw,x,x+xw)
        img1new = cv2.resize(imgrgb,dz2,interpolation=cv2.INTER_LINEAR)
        #cv2.imshow("thernew",img2new)
        if img2n is not None and img2n.size > 0:
            imgnew = cv2.applyColorMap(img2new,11)
            imgnew = cv2.warpPerspective(imgnew,h,img2new.shape[:2][::-1])
            imgnew = cv2.resize(imgnew,dz2,interpolation=cv2.INTER_LINEAR)
            img_temp = np.copy(imgnew)
            #imgmerge=recognize_shapes(imgrgb)
            if index == 0:
                mask, boolindex = recognize_shapes(imgrgb)
                if boolindex == 1:
                    maskfinal = mask
                    index = 1
            if index == 1:
                tempimg = cv2.bitwise_and(img_temp,img_temp,mask=mask)       
                imgmerge1 = cv2.add(tempimg,img1n)
                #cv2.imshow('img_temp', img2new)
                #cv2.imshow('tempimg', tempimg)
                #cv2.imshow('merge', imgmerge1)
            #Write the frame to the video file
                out.write(imgmerge1)
                cv2.imshow('merge', imgmerge1)
        else:
            print("imgthermal[y:y+yw,x:x+xw] is None")


        detector = aruco.ArucoDetector(dictionary,parameters)
        frame = img1n
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)



        circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=30, maxRadius=80)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Sort circles by radius in descending order and select the top 9
            circles = sorted(circles, key=lambda x: x[2], reverse=True)[:circle_number]
            # print(f"Number of circles detected: {len(circles)}")
            roicircle = circles
            # Prepare data for publishing
            circle_data = Float64MultiArray()
            circle_data.data = []
            for (x, y, r) in circles:
                cv2.circle(img1n, (x, y), r, (0, 255, 0), 4)
                # cv2.rectangle(img1n, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                #Append circle data to the array
                circle_data.data.extend([x, y, r])
                
        #else:
            #print("No circles detected")
   

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
        img1n=aruco.drawDetectedMarkers(img1n, markerCorners, markerIds)
        
        if(markerIds is not None):
            print("length of markerIds",len(markerIds))
            
            corners = markerCorners
            
            
            # print(corners)
            ARUIDS = markerIds
            pose_array = create_robot_pose_array(corners,markerIds)
            print(pose_array)
            pub.publish(pose_array)
            if len(triangle_data) == 0 or len(rectangle_data) == 0:
                continue
            else:
                publisher()
             # Plot the direction of the ArUco markers

            corner = markerCorners[0][0]
            center = np.mean(corner, axis=0).astype(int)
            direction_angle = math.atan2(corner[0][1] - corner[3][1], corner[0][0] - corner[3][0])
            direction_length = 50  # Length of the direction arrow
            end_point = (int(center[0] + direction_length * math.cos(direction_angle)),
                         int(center[1] + direction_length * math.sin(direction_angle)))
            cv2.arrowedLine(img1n, tuple(center), end_point, (255, 0, 0), 2, tipLength=0.3)

        # cv2.imshow("ther",img2n)
        # img2n = imgthermal[y:y+yw,x:x+xw]
        # img1n = cv2.resize(imgrgb,dz2,interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("rgbnew",img1n)

        # imgnew = cv2.applyColorMap(img2n,11)
        # imgnew = cv2.warpPerspective(imgnew,h,img2n.shape[:2][::-1])
        # imgnew = cv2.resize(imgnew,dz2,interpolation=cv2.INTER_LINEAR)

        # img_temp = np.copy(imgnew)

        # imgmerge=recognize_shapes(imgrgb)
        
        if img1n is not None and img1n.size > 0:
            cv2.imshow("Detected Circles", img1n)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
           
out.release()
videorgb.release()
videothermal.release()
cv2.destroyAllWindows()
