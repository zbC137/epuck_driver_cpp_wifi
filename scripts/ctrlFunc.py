#!/usr/bin/env python3

import numpy.matlib
import math
import time

import numpy as np
import matplotlib.pyplot as plt


def curve(t, s):
    '''
    # flower curve
    x = (6 + 0.1 * t) * (20 - 20 ** (np.sin(5 * 2 * math.pi * s))) * np.cos(2 * math.pi * s) + 600
    y = (6 + 0.1 * t) * (20 - 20 ** (np.sin(5 * 2 * math.pi * s))) * np.sin(2 * math.pi * s) + 300
    '''
    '''
    # star curve
    x = (3*t+(0.5*t*np.sin(4*math.pi*s)+t*np.cos(10*math.pi*s)+150)*np.cos(2*math.pi*s)) + 400
    y = (0.5*t*np.sin(4*math.pi*s)+t*np.cos(10*math.pi*s)+150)*np.sin(2*math.pi*s) + 250
    '''
    '''
    if t < 30:
        x = 200 * np.cos(2*math.pi*s) + 350
        y = 200 * np.sin(2*math.pi*s) + 300
    elif t<60:
        x = 200 * np.cos(2 * math.pi * s) + 500
        y = 150 * np.sin(2 * math.pi * s) + 300
    else:
        x = 100 + (150 * (1 - np.sin(2 * math.pi * s))) * np.cos(2 * math.pi * s) + 700
        y = 100 + (150 * (1 - np.sin(2 * math.pi * s))) * np.sin(2 * math.pi * s) + 350
    '''
    '''
    if t < 60:
        x = 100 + (150 * (1 - np.sin(2 * math.pi * s))) * np.cos(2 * math.pi * s) + 400
        y = 100 + (150 * (1 - np.sin(2 * math.pi * s))) * np.sin(2 * math.pi * s) + 350
    else:
        x = (20 * np.sin(4 * math.pi * s) + 40 * np.cos(10 * math.pi * s) + 150) * np.cos(2 * math.pi * s) + 750
        y = (20 * np.sin(4 * math.pi * s) + 40 * np.cos(10 * math.pi * s) + 150) * np.sin(2 * math.pi * s) + 250
    '''
    # open curve
    x = (1 - s) ** 3 * 350 - 3 * s * (1 - s) ** 2 * 50 - 3 * s ** 2 * (1 - s) * 200 - s ** 3 * 200 + 600
    y = (1 - s) ** 3 * 300 - 3 * s * (1 - s) ** 2 * 400 + 3 * s ** 2 * (1 - s) * 600 - s ** 3 * 100 + 175
    '''
    # curve for paper
    x = (3 * t + (0.5*t * np.sin(4 * math.pi * s) + 200) * np.cos(2 * math.pi * s)) +300
    y = ((0.5*t * np.cos(4 * math.pi * s) + 200) * np.sin(2 * math.pi * s))+300
    '''

    '''
    # elipse
    x = 200 * (1 + 0.01 * t) * np.cos(2 * math.pi * s) + 400 + 3 * t
    y = 100 * (1 + 0.01 * t) * np.sin(2 * math.pi * s) + 300
    '''
    return x, y


def cLength(t):
    s = np.arange(1001).reshape(1001, 1) * 0.001
    lenList = np.matlib.zeros((1001, 1))

    x, y = curve(t, s)

    lenList[1:] = np.matlib.square(x[1:] - x[:-1]) + np.matlib.square(y[1:] - y[:-1])
    lenList = np.matlib.sqrt(lenList)
    length = sum(lenList)

    return lenList / length


def objLength(objX, objY, nPt):
    lenList = np.matlib.zeros((nPt, 1))
    lenList[1:] = (
            np.matlib.square(objX[0, 1:] - objX[0, :-1]) + np.matlib.square(objY[0, 1:] - objY[0, :-1])).transpose()
    lenList = np.matlib.sqrt(lenList)
    length = sum(lenList)

    return lenList / length


def generateX(values, agentNum):
    x = np.matlib.zeros((3 * agentNum, 1))
    for i in range(agentNum):
        x[2 * i] = values[3 * i]
        x[2 * i + 1] = values[3 * i + 1]
        x[2 * agentNum + i] = values[3 * i + 2]

    return x


def generateGs(start, step, boundary, t, flag, harmNum):
    if flag == 0:
        n = (boundary - step / 2 - start) // step + 1
    else:
        lenList = cLength(t)
        n = len(lenList)
    gs = np.matlib.zeros((2 * int(n), 4 * harmNum + 2))
    s = 0
    for i in range(int(n)):
        if flag == 0:
            s = start + i * step + 0.5 * step
        else:
            s += lenList[i]

        for j in range(harmNum):
            gs[2 * i, 4 * j] = math.cos(2 * math.pi * (j + 1) * s)
            gs[2 * i + 1, 4 * j + 2] = math.cos(2 * math.pi * (j + 1) * s)
            gs[2 * i, 4 * j + 1] = math.sin(2 * math.pi * (j + 1) * s)
            gs[2 * i + 1, 4 * j + 3] = math.sin(2 * math.pi * (j + 1) * s)

        gs[2 * i, 4 * harmNum], gs[2 * i + 1, 4 * harmNum + 1] = 1, 1

    return gs


def generateGsOpen(start, step, boundary, t, flag, harmNum):
    if flag == 0:
        n = (boundary - step / 2 - start) // step + 1
    else:
        lenList = cLength(t)
        n = len(lenList)
    gs = np.matlib.zeros((int(n), harmNum))
    s = 0
    for i in range(int(n)):
        if flag == 0:
            s = start + i * step + 0.5 * step
        else:
            s += lenList[i]

        for j in range(harmNum):
            gs[i, j] = s ** j
    I = np.matlib.identity(2)

    return np.matlib.kron(gs, I)


def generateGsNew(start, step, boundary, t, flag, harmNum, objX, objY, nPt):
    if flag == 0:
        n = (boundary - step / 2 - start) // step + 1
    else:
        lenList = objLength(objX, objY, nPt)
        n = len(lenList)
    gs = np.matlib.zeros((2 * int(n), 4 * harmNum + 2))
    s = 0
    for i in range(int(n)):
        if flag == 0:
            s = start + i * step
        else:
            s += lenList[i]

        for j in range(harmNum):
            gs[2 * i, 4 * j] = math.cos(2 * math.pi * (j + 1) * s)
            gs[2 * i + 1, 4 * j + 2] = math.cos(2 * math.pi * (j + 1) * s)
            gs[2 * i, 4 * j + 1] = math.sin(2 * math.pi * (j + 1) * s)
            gs[2 * i + 1, 4 * j + 3] = math.sin(2 * math.pi * (j + 1) * s)

        gs[2 * i, 4 * harmNum], gs[2 * i + 1, 4 * harmNum + 1] = 1, 1

    return gs


def generateL(k, agentNum):
    L = np.matlib.identity(agentNum)

    if k > 0:
        L *= 2 * k
        for i in range(agentNum):
            for j in range(i - k + 1, i + k + 2):
                if j == i + 1:
                    continue
                elif j <= 0:
                    temp = agentNum + j
                elif j > agentNum:
                    temp = j - agentNum
                else:
                    temp = j

                L[i, temp - 1] = -1

    I = np.matlib.identity(2)

    return np.matlib.kron(L, I)


def newGenerateL(k, agentNum):
    L = np.matlib.identity(agentNum)

    for i in range(agentNum):
        if i == k - 1:
            L[i, i] = 0
        elif i < k - 1:
            L[i, i + 1] = -1
        else:
            L[i, i - 1] = -1

    I = np.matlib.identity(2)

    return np.matlib.kron(L, I)


def curveFitting(gs, cd, harmNum):
    #coff = np.matlib.ones((4 * harmNum + 2, 1))
    coff = np.matlib.ones((2*harmNum, 1))

    nDelta = 1e+4
    fs = gs * coff
    err = fs - cd

    while nDelta > 1e-4:
        delta = -np.linalg.inv(np.matlib.transpose(gs) * gs) * np.matlib.transpose(gs) * err
        coff += delta
        fs = gs * coff
        err = fs - cd
        nDelta = np.linalg.norm(delta)

    return coff


def objFitting(gs, cd, harmNum):
    coff = np.matlib.ones((4 * harmNum + 2, 1))

    nDelta = 1e+4
    fs = gs * coff
    err = fs - cd

    while nDelta > 1e-4:
        delta = -np.linalg.inv(np.matlib.transpose(gs) * gs) * np.matlib.transpose(gs) * err
        coff += delta
        fs = gs * coff
        err = fs - cd
        nDelta = np.linalg.norm(delta)

    return coff


def generateR(dist, angles, agentNum):
    R = np.matlib.zeros((2 * agentNum, 2 * agentNum))

    for i in range(agentNum):
        R[2 * i, 2 * i] = np.cos(angles[i])
        R[2 * i, 2 * i + 1] = -dist * np.sin(angles[i])
        R[2 * i + 1, 2 * i] = np.sin(angles[i])
        R[2 * i + 1, 2 * i + 1] = dist * np.cos(angles[i])

    return R


def controller(t, x, L, ctrlK, dist, gc, f, agentNum, harmNum):
    posx = x[0:2 * agentNum - 1:2] + dist * np.cos(x[2 * agentNum:])
    posy = x[1:2 * agentNum:2] + dist * np.sin(x[2 * agentNum:])
    pos = np.hstack((posx, posy)).reshape(2 * agentNum, 1)
    R = generateR(dist, x[2 * agentNum:], agentNum)

    s = np.arange(1001).reshape(1001, 1) * 0.001
    x, y = curve(t, s)
    cd = np.hstack((x, y)).reshape(2002, 1)
    gs = generateGs(0, 0.001, 1, t, 1, harmNum)
    coff = curveFitting(gs, cd, harmNum)
    coffErr = np.linalg.pinv(gc) * pos - coff
    posErr = pos - gc * coff
    xi = (np.linalg.pinv(gc) * gc - np.matlib.identity(4 * harmNum + 2)) * coff

    u1 = - ctrlK[0] * L * gc * coffErr - ctrlK[1] * posErr
    u2 = np.matmul(np.linalg.inv(R), u1)

    return u2, coffErr, posErr, xi


def newController(t, x, L, ctrlK, dist, gc, f, agentNum, harmNum, errCum, d):
    posx = x[0:2 * agentNum - 1:2] + dist * np.cos(x[2 * agentNum:])
    posy = x[1:2 * agentNum:2] + dist * np.sin(x[2 * agentNum:])
    pos = np.hstack((posx, posy)).reshape(2 * agentNum, 1)
    R = generateR(dist, x[2 * agentNum:], agentNum)

    s = np.arange(1001).reshape(1001, 1) * 0.001
    x, y = curve(t, s)
    cd = np.hstack((x, y)).reshape(2002, 1)
    # gs = generateGs(0, 0.001, 1, t, 1, harmNum)
    gs = generateGsOpen(0, 0.001, 1, t, 1, harmNum)
    coff = curveFitting(gs, cd, harmNum)
    coffErr = np.linalg.pinv(gc) * pos - coff
    posErr = pos - gc * coff
    errCum += ctrlK[1] * 1 / f * R.transpose() * posErr
    posErrLeader = np.matlib.zeros((2 * agentNum, 1))
    posErrLeader[8, 0] = posErr[8, 0]
    posErrLeader[9, 0] = posErr[9, 0]
    #xi = (np.linalg.pinv(gc) * gc - np.matlib.identity(4 * harmNum + 2)) * coff
    xi = (np.linalg.pinv(gc) * gc - np.matlib.identity(2*harmNum)) * coff

    u1 = - ctrlK[0] * (L * gc * coffErr + posErrLeader) - ctrlK[1] * R * errCum + R * d
    u2 = np.matmul(np.linalg.inv(R), u1)

    return u2, coffErr, posErr, xi, errCum, gs, coff


def objController(t, x, L, ctrlK, dist, gc, f, agentNum, harmNum, errCum, d, cloudData):
    posx = x[2:2 * agentNum + 1:2] + dist * np.cos(x[2 * agentNum + 3:])
    posy = x[3:2 * agentNum + 2:2] + dist * np.sin(x[2 * agentNum + 3:])
    pos = np.hstack((posx, posy)).reshape(2 * agentNum, 1)
    R = generateR(dist, x[2 * agentNum + 3:], agentNum)

    objPosX, objPosY = x[0] * 1.5037594e-3, x[1] * 1.5306122e-3
    # objPosX, objPosY = x[0], x[1]
    cloudPolar = cloudData[0:, 2] + 0.08
    cloudAngle = cloudData[0:, 3] + (x[2 * agentNum + 2] - 0.0499584)
    objX = (objPosX + np.array([cloudPolar]) * np.array(np.cos(cloudAngle))) / 1.5037594e-3
    objY = (objPosY + np.array([cloudPolar]) * np.array(np.sin(cloudAngle))) / 1.5306122e-3
    cd = np.empty((objX.size + objY.size, 1), dtype=objX.dtype)
    cd[1::2, 0] = objY
    cd[::2, 0] = objX
    # cd = np.hstack((objX, objY)).reshape(2*len(cloudPolar), 1)
    '''
    print(cd)
    print(objX)
    print(objY)
    '''
    gs = generateGsNew(0, 0.001, 1, t, 1, harmNum, objX, objY, len(cloudPolar))
    coff = objFitting(gs, np.array(cd), harmNum)
    coffErr = np.linalg.pinv(gc) * pos - coff
    posErr = pos - gc * coff
    errCum += ctrlK[1] * 1 / f * R.transpose() * posErr
    posErrLeader = np.matlib.zeros((2 * agentNum, 1))
    posErrLeader[8, 0] = posErr[8, 0]
    posErrLeader[9, 0] = posErr[9, 0]
    xi = (np.linalg.pinv(gc) * gc - np.matlib.identity(4 * harmNum + 2)) * coff

    u1 = - ctrlK[0] * (L * gc * coffErr + posErrLeader) - ctrlK[1] * R * errCum + R * d
    u2 = np.matmul(np.linalg.inv(R), u1)

    return u2, coffErr, posErr, xi, errCum, gs * coff, coff, objX, objY
