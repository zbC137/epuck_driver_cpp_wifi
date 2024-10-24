#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray


class Robot:

    def __init__(self, botNum):
        self.botNum = botNum
        self.botPose = np.matlib.zeros((3 * self.botNum, 1))
        self.botId = 0
        self.sub = rospy.Subscriber('robot_pose', Float64MultiArray, self.state_callback, queue_size=10)

    def state_callback(self, msg):
        n = self.botNum
        for i in range(n):
            self.botId = int(msg.data[4*i + 3])
            id = self.botId - 1
            self.botPose[3*id:3*id+3] = np.array([msg.data[4*i], msg.data[4*i + 1], msg.data[4*i + 2]]).reshape((3, 1))
