#!/usr/bin/env python3
import time
import rospy
import ctrlFunc as cf
import numpy as np
import matplotlib.pyplot as plt
import csv

from robot import Robot
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

agentNum = 8
harmNum = 6
#ctrlK = [1e-3, 3e-3]
ctrlK = [2/3, 2]
k = 2
dist = 0.001
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
        
        t_pub = rospy.Publisher('time', Float64MultiArray, queue_size=10)
        
        pub_1 = rospy.Publisher('epuck_robot_0/vel', Float64MultiArray, queue_size=10)
        pub_2 = rospy.Publisher('epuck_robot_1/vel', Float64MultiArray, queue_size=10)
        pub_3 = rospy.Publisher('epuck_robot_2/vel', Float64MultiArray, queue_size=10)
        pub_4 = rospy.Publisher('epuck_robot_3/vel', Float64MultiArray, queue_size=10)
        pub_5 = rospy.Publisher('epuck_robot_4/vel', Float64MultiArray, queue_size=10)
        pub_6 = rospy.Publisher('epuck_robot_5/vel', Float64MultiArray, queue_size=10)
        pub_7 = rospy.Publisher('epuck_robot_6/vel', Float64MultiArray, queue_size=10)
        pub_8 = rospy.Publisher('epuck_robot_7/vel', Float64MultiArray, queue_size=10)
        #pub_9 = rospy.Publisher('epuck_robot_8/vel', Float64MultiArray, queue_size=10)
        
        vel = [0] * 2 * agentNum
        ce = [0] * (4 * harmNum + 2)
        pe = [0] * (2 * agentNum)
        ori = [0] * agentNum
        xi_m = [0]*(4 * harmNum + 2)
        rate = rospy.Rate(f)

        bot_state = Robot(agentNum)
        l = cf.generateL(k, agentNum)
        gc = cf.generateGs(0, 1 / agentNum, 1, 0, 0, harmNum)
        start_time = time.time()
        t1 = 0

        while not rospy.is_shutdown():
            t = time.time() - start_time
            u, coffErr, posErr, xi = cf.controller(t, bot_state.botPose, l, ctrlK, dist, gc, f, agentNum, harmNum)
            u = np.zeros((2 * agentNum, 1))
            if t > 75:
                v_com = 0
            elif t > 40:
                v_com = -0.003
            elif t > 35:
                v_com = 0
            else:
                v_com = 0.003
                
            for i in range(agentNum):
                vel[2 * i] = u[2 * i, 0] + v_com
                vel[2 * i + 1] = u[2 * i + 1, 0]
                
                pe[2 * i] = posErr[2 * i, 0]
                pe[2 * i + 1] = posErr[2 * i + 1, 0]

                ori[i] = bot_state.botPose[3*i + 2]

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

            t_msg = Float64MultiArray(data=[t])
            rospy.loginfo(t_msg)
            t_pub.publish(t_msg)
            
            vel_msg = Float64MultiArray(data=vel)
            rospy.loginfo(vel_msg)
            
            pub_1.publish(Float64MultiArray(data=vel[0:2]))
            pub_2.publish(Float64MultiArray(data=vel[2:4]))
            pub_3.publish(Float64MultiArray(data=vel[4:6]))
            pub_4.publish(Float64MultiArray(data=vel[6:8]))
            pub_5.publish(Float64MultiArray(data=vel[8:10]))
            pub_6.publish(Float64MultiArray(data=vel[10:12]))
            pub_7.publish(Float64MultiArray(data=vel[12:14]))
            pub_8.publish(Float64MultiArray(data=vel[14:16]))
            #pub_9.publish(Float64MultiArray(data=vel[16:18]))

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