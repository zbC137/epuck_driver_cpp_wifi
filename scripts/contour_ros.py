#!/usr/bin/env python3
import time
import rospy
import math
import ctrlFunc as cf
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.io as sio

from multiprocessing import Process
from robot import Robot
from std_msgs.msg import Float64MultiArray

# mat = sio.loadmat('/home/romi/objdata/cloudData.mat')
# cloudData = mat['cloudData']
# mat = sio.loadmat('/home/romi/objdata/Circle/circleData.mat')
# cloudData = mat['circleData']
mat = sio.loadmat('/home/romi/objdata/Square/squareData.mat')
cloudData = mat['squareData']
cloudPose = cloudData[0:, 0:2]
cloudPolar = cloudData[0:, 2]
nPt = len(cloudPolar)

# agentDict = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '11': 7, '12': 8, '13': 9}
agentDict = {'4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '11': 7, '12': 8, '13': 9}
agentNum = len(agentDict) #- 1
harmNum = 10
ctrlK = [1e-3, 5e-4]
# ctrlK = [2/3, 1/3]
k = 5
dist = -0.01

f = 5

wheel_base = 80e-3  # m
wheel_diameter = 31e-3  # m

linear_sat = 0.02
angular_sat = 0.1

MAX_SPEED = 1.5


def saturated(input, bound):
    if input > bound:
        input = bound
    elif input < -bound:
        input = -bound

    return input


if __name__ == '__main__':
    try:
        rospy.init_node('fouriCtrl', anonymous=True)
        pub = rospy.Publisher('anglevelocity', Float64MultiArray, queue_size=10)
        # pub1 = rospy.Publisher('obj', Float64MultiArray, queue_size=10)
        vel = [0] * (3 * agentNum + 1)
        ce = [0] * (4 * harmNum + 2)
        pe = [0] * (2 * agentNum)
        ori = [0] * agentNum
        xi_m = [0] * (4 * harmNum + 2)
        rate = rospy.Rate(f)
        errCum = np.matlib.ones((2 * agentNum, 1))
        d = np.matlib.zeros((2 * agentNum, 1)) + 0.005
        ec = [0] * (2 * agentNum)
        obj = [0] * (4 * nPt + 401)
        ind = 0

        bot_state = Robot(agentNum, agentDict)
        l = cf.newGenerateL(5, agentNum)
        #gc = cf.generateGs(0, 1 / agentNum, 1, 0, 0, harmNum)
        gc = cf.generateGsOpen(0, 1 / agentNum, 1, 0, 0, harmNum)
        gg = cf.generateGs(0, 0.005, 1, 0, 0, harmNum)
        start_time = time.time()

        while not rospy.is_shutdown():
            ind += 1
            t = time.time() - start_time
            # u, coffErr, posErr, xi, errCum, objFit, coff, offsetX, offsetY = cf.objController(t, bot_state.x, l, ctrlK,
            #                                                                                  dist, gc, f,
            #                                                                                  agentNum, harmNum,
            #                                                                                  errCum, d, cloudData)
            u, coffErr, posErr, xi, errCum, gs, coff = cf.newController(t, bot_state.x, l, ctrlK, dist, gc, f,
                                                              agentNum, harmNum,
                                                              errCum, d)
            for i in range(agentNum):
                left = (u[2 * i, 0] - u[2 * i + 1, 0] * wheel_base / 2) * 2 / wheel_diameter
                right = (u[2 * i, 0] + u[2 * i + 1, 0] * wheel_base / 2) * 2 / wheel_diameter

                vel[3 * i] = saturated(left, MAX_SPEED)
                vel[3 * i + 1] = saturated(right, MAX_SPEED)

                pe[2 * i] = posErr[2 * i, 0]
                pe[2 * i + 1] = posErr[2 * i + 1, 0]
                vel[3 * i + 2] = np.sqrt(np.square(pe[2 * i] * 1.5037594e-3) + np.square(pe[2 * i + 1] * 1.5306122e-3))

                ec[2 * i] = errCum[2 * i, 0]
                ec[2 * i + 1] = errCum[2 * i + 1, 0]

                ori[i] = bot_state.x[2 * agentNum + i, 0]
            '''
            for i in range(harmNum):
                ce[4 * i] = coffErr[4 * i, 0]
                ce[4 * i + 1] = coffErr[4 * i + 1, 0]
                ce[4 * i + 2] = coffErr[4 * i + 2, 0]
                ce[4 * i + 3] = coffErr[4 * i + 3, 0]

                xi_m[4 * i] = xi[4 * i, 0]
                xi_m[4 * i + 1] = xi[4 * i + 1, 0]
                xi_m[4 * i + 2] = xi[4 * i + 2, 0]
                xi_m[4 * i + 3] = xi[4 * i + 3, 0]

            ce[4 * harmNum] = coffErr[4 * harmNum, 0]
            ce[4 * harmNum + 1] = coffErr[4 * harmNum + 1, 0]

            xi_m[4 * harmNum] = xi[4 * harmNum, 0]
            xi_m[4 * harmNum + 1] = xi[4 * harmNum + 1, 0]
            '''
            '''
            estPos = gg * coff
            print(len(estPos))
            objPosX, objPosY = bot_state.x[0] * 1.5037594e-3, bot_state.x[1] * 1.5306122e-3
            # objPosX, objPosY = x[0], x[1]
            # cloudPolar = cloudData[0:, 2] + 0.08
            cloudAngle = cloudData[0:, 3] + (bot_state.x[2 * agentNum + 2] - 0.0)
            objX = (objPosX + np.array([cloudPolar]) * np.array(np.cos(cloudAngle))) / 1.5037594e-3
            objY = (objPosY + np.array([cloudPolar]) * np.array(np.sin(cloudAngle))) / 1.5306122e-3

            obj[0] = ind
            obj[1:nPt + 1] = objX.flat
            obj[nPt + 1:2 * nPt + 1] = objY.flat
            obj[2 * nPt + 1: 3 * nPt + 1] = offsetX.flat
            obj[3 * nPt + 1:4 * nPt + 1] = offsetY.flat
            obj[4 * nPt + 1:] = estPos
            obj_msg = Float64MultiArray(data=obj)
            pub1.publish(obj_msg)
            '''
            vel[-1] = t
            vel_msg = Float64MultiArray(data=vel)
            rospy.loginfo(vel_msg)
            pub.publish(vel_msg)
            t1 = time.time() - start_time
            print("t: ", t1 - t)

            fp = open('data.csv', mode='a+', newline='')
            dp = csv.writer(fp)
            #dp.writerow(vel + ce + pe + ori + xi_m + ec)
            dp.writerow(vel + pe + ori + ec)
            fp.close()
            '''
            posx = bot_state.x[0:2 * agentNum + 1:2] + dist * np.cos(bot_state.x[2 * agentNum + 2:])
            posy = bot_state.x[1:2 * agentNum + 2:2] + dist * np.sin(bot_state.x[2 * agentNum + 2:])
            # s = np.arange(1001).reshape(1001, 1) * 0.001
            # x, y = cf.curve(t, s)

            # s_a = np.arange(agentNum).reshape(agentNum, 1) / agentNum
            # px, py = cf.curve(t, s_a)
            plt.plot(objX.flat, objY.flat, color='r')
            # plt.plot(objFit[0:2 * nPt:2], objFit[1:2 * nPt:2], color='b')
            plt.plot(estPos[0::2], estPos[1::2], color='b')
            colors = ["green", "black", "orange", "purple", "blue", "red", "cyan", "yellow", "magenta", "green"]
            # plt.scatter(px, py, marker="*", c=colors)
            plt.scatter(posx.flat, posy.flat, c=colors)

            # plt.scatter(t, e*1.5e-3, color='b')
            plt.pause(1 / f)
            plt.ioff()
            plt.cla()
            '''
            s = np.arange(1001).reshape(1001, 1) * 0.001
            x, y = cf.curve(t, s)
            pos = gs*coff
            plt.plot(x, y, color='r')
            plt.plot(pos[0::2], pos[1::2], color='b')
            plt.pause(1 / f)
            plt.ioff()
            plt.cla()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
