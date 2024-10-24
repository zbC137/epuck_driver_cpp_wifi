#!/usr/bin/env python3
import time
import rospy
import ctrlFunc as cf
import numpy as np
import matplotlib.pyplot as plt
import csv

from robot import Robot
from std_msgs.msg import Float64MultiArray

agentNum = 9
harmNum = 6
ctrlK = [1e-3, 3e-3]
k = 2
dist = -0.01
f = 10

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
        pub = rospy.Publisher('vel', Float64MultiArray, queue_size=10)
        vel = [0] * (2 * agentNum + 1)
        ce = [0] * (4 * harmNum + 2)
        pe = [0] * (2 * agentNum)
        ori = [0] * agentNum
        xi_m = [0]*(4 * harmNum + 2)
        rate = rospy.Rate(f)

        bot_state = Robot(agentNum)
        l = cf.generateL(k, agentNum)
        gc = cf.generateGs(0, 1 / agentNum, 1, 0, 0, harmNum)
        start_time = time.time()

        while not rospy.is_shutdown():
            t = time.time() - start_time
            u, coffErr, posErr, xi = cf.controller(t, bot_state.botPose, l, ctrlK, dist, gc, f, agentNum, harmNum)

            for i in range(agentNum):
                vel[2 * i] = u[2 * i, 0]
                vel[2 * i + 1] = u[2 * i + 1, 0]

                pe[2 * i] = posErr[2 * i, 0]
                pe[2 * i + 1] = posErr[2 * i + 1, 0]

                ori[i] = bot_state.x[2 * agentNum + i,0]

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

            vel[-1] = t
            vel_msg = Float64MultiArray(data=vel)
            rospy.loginfo(vel_msg)
            pub.publish(vel_msg)
            t1 = time.time() - start_time
            print("t: ", t1 - t)

            fp = open('data.csv', mode='a+', newline='')
            dp = csv.writer(fp)
            dp.writerow(vel + ce + pe + ori + xi_m)
            fp.close()

            '''
            posx = bot_state.x[0:2 * agentNum - 1:2] + dist * np.cos(bot_state.x[2 * agentNum:])
            posy = bot_state.x[1:2 * agentNum:2] + dist * np.sin(bot_state.x[2 * agentNum:])
            s = np.arange(1001).reshape(1001, 1) * 0.001
            x, y = cf.curve(t, s)

            s_a = np.arange(agentNum).reshape(agentNum, 1) / agentNum
            px, py = cf.curve(t, s_a)
            
            plt.plot(x, y, color='r')
            colors = ["green", "black", "orange", "purple", "blue", "red", "cyan"]
            plt.scatter(px, py, marker="*", c=colors)
            plt.scatter(posx.flat, posy.flat, c=colors)
            
            #plt.scatter(t, e*1.5e-3, color='b')
            plt.pause(1 / f)
            plt.ioff()
            plt.cla()
            '''
            rate.sleep()

    except rospy.ROSInterruptException:
        pass