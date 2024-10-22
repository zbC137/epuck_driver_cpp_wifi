#!/usr/bin/env python3

import rospy
import numpy.matlib
import numpy as np
from robot_msg.msg import robot_pose_array


class PID:

    def __init__(self, newError, samplingPeriod):
        self.currError = newError
        self.lastError = 0
        self.errorInt = newError * samplingPeriod

    def proportional(self, newError, kp):
        up = kp * newError

        return up

    def integral(self, newError, ki, samplingPeriod):
        self.errorInt += newError * samplingPeriod
        ui = ki * self.errorInt

        return ui

    def derivative(self, newError, kd, samplingPeriod):
        ud = kd * (newError - self.lastError) / samplingPeriod
        self.lastError = newError

        return ud

    def ctrl_pid(self, newError, kp, ki, kd, samplingPeriod):
        up = self.proportional(newError, kp)
        ui = self.integral(newError, ki, samplingPeriod)
        ud = self.derivative(newError, kd, samplingPeriod)

        return up + ui + ud

    def ctrl_pi(self, newError, kp, ki, samplingPeriod):
        up = self.proportional(newError, kp)
        ui = self.integral(newError, ki, samplingPeriod)

        return up + ui

    def ctrl_pd(self, newError, kp, kd, samplingPeriod):
        up = self.proportional(newError, kp)
        ud = self.derivative(newError, kd, samplingPeriod)

        return up + ud
