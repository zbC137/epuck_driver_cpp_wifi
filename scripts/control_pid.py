#!/usr/bin/env python3
import time
import rospy
import math
import ctrlFunc as cf
import numpy as np
import matplotlib.pyplot as plt

from robot import Robot
from mypid import PID
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

agentNum = 1
harmNum = 6
ctrlK = 2
k = 2

f = 10

wheel_base = 80e-3  # mm
wheel_diameter = 31e-3  # mm

if __name__ == '__main__':
    try:
        rospy.init_node('fouriCtrl', anonymous=True)

        pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        vel = Twist()
        rate = rospy.Rate(f)

        bot_state = Robot(agentNum)
        linPID = PID(0, 1 / f)
        angPID = PID(0, 1 / f)

        # parameters
        kp_l, ki_l, kd_l = 0.6, 0.0, 0.0
        kp_a, ki_a, kd_a = 0.5, 0.0, 0.0

        target = [0.8, 0.35, math.pi / 4]

        start_time = time.time()
        frame_count = 0

        while not rospy.is_shutdown():
            t = time.time() - start_time

            angle = frame_count * 2 * math.pi / 1000
            xd = 0.5 * np.cos(angle)
            yd = 0.25 * np.sin(angle)

            for i in range(agentNum):
                pos = bot_state.x.flat
                posError = math.sqrt((xd - pos[0]) ** 2 + (yd - pos[1]) ** 2)
                angError = math.atan2((yd - pos[1]), (xd - pos[0])) - pos[2]
                if angError > math.pi:
                    angError -= 2 * math.pi
                elif angError < -math.pi:
                    angError += 2 * math.pi

                uLinear = linPID.ctrl_pid(posError, kp_l, ki_l, kd_l, 1 / f)
                uAngular = -angPID.ctrl_pid(angError, kp_a, ki_a, kd_a, 1 / f)

                #vel[2 * i] = (uLinear - uAngular * wheel_base / 2) * 2 / wheel_diameter
                #vel[2 * i + 1] = (uLinear + uAngular * wheel_base / 2) * 2 / wheel_diameter

                frame_count += 1

                plt.scatter(xd, yd, marker="*", color='r')
                # plt.scatter(target[0], target[1], marker="*", color='r')
                plt.scatter(pos[0], pos[1], color='b')
                plt.pause(1 / 30)
                plt.ioff()

                vel.angular = uAngular
                vel.linear = uLinear
                #vel_msg = Float64MultiArray(data=vel)
                rospy.loginfo(vel)
                pub.publish(vel)

                rate.sleep()

    except rospy.ROSInterruptException:
        pass
