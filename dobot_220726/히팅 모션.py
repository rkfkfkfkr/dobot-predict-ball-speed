import cv2
import numpy as np
import imutils
import threading
import math
import time
import DobotDllType as dType
from numpy.linalg import inv
import matplotlib.pyplot as plt

import cv2.aruco as aruco
import os

from multiprocessing import Process, Pipe, Queue, Value, Array, Lock

api = dType.load()

def dobot_get_start():
    
    dType.SetQueuedCmdClear(api)

    dType.SetHOMEParams(api, 200, 0, 20, -10, 0) # x, y, z, r 

    dType.SetJOGJointParams(api, 900, 700, 700,480,300, 300, 300, 300, 0)
    dType.SetJOGCoordinateParams(api,900,900,900,480,300,300,300,300,0)
    dType.SetJOGCommonParams(api, 900,350,0)

    dType.SetPTPJointParams(api,320,320,320,480,300,300,300,300,0) # velocity[4], acceleration[4]
    dType.SetPTPCoordinateParams(api,700,700,480,300,0) 
    dType.SetPTPCommonParams(api, 500, 350,0) # velocityRatio(속도율), accelerationRation(가속율)
       
    dType.SetHOMECmd(api, temp = 0, isQueued = 0)

    #dType.SetQueuedCmdStartExec(api)

def dobot_connect():

    CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound", 
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):
        dobot_get_start()

def segmentaition(frame):

    img_ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(img_ycrcb)

    _, cb_th = cv2.threshold(cb, 90, 255, cv2.THRESH_BINARY_INV)
    cb_th = cv2.dilate(cv2.erode(cb_th, None, iterations=2), None, iterations=2)
    #cb_th = cv2.dilate(cb_th, None, iterations=2)

    return cb_th

def get_distance(x, y, imagePoints):
    
    objectPoints = np.array([[33.6,85,0],
                            [33.6,75,0],
                            [23.6,75,0],
                            [23.6,85,0],],dtype = 'float32')


    fx = float(470.5961)
    fy = float(418.18176)
    cx = float(275.7626)
    cy = float(240.41246)
    k1 = float(0.06950)
    k2 = float(-0.07445)
    p1 = float(-0.01089)
    p2 = float(-0.01516)

    #cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float64')
    #distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')

    cameraMatrix = np.array([[470.5961,0,275.7626],[0,418.18176,240.41246],[0,0,1]],dtype = 'float32')
    distCoeffs = np.array([0.06950,-0.07445,-0.01089,-0.01516],dtype = 'float32')
    _,rvec,t = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    R,_ = cv2.Rodrigues(rvec)
            
    u = (x - cx) / fx
    v = (y - cy) / fy
    Qc = np.array([[u],[v],[1]])
    Cc = np.zeros((3,1))
    Rt = np.transpose(R)
    Qw = Rt.dot((Qc-t))
    Cw = Rt.dot((Cc-t))
    V = Qw - Cw
    k = -Cw[-1,0]/V[-1,0]
    Pw = Cw + k*V
    
    px = Pw[0]
    py = Pw[1]

    #print("px: %f, py: %f" %(px,py))

    return px,py

def find_ball(frame,cb_th,box_points):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    px = None
    py = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 5:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
            px,py = get_distance(center[0], center[1],box_points)
            
            #text = " %f , %f" %(px,py)
            #cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            #print("px: %f, py: %f" %(px,py))

    return px,py

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if draw:
        cv2.aruco.drawDetectedMarkers(img,bboxs)
        #print(len(bboxs))

    if len(bboxs) > 0:
        return bboxs[0][0]
    else:
        return [0,0]

def hit():

    #dobot_connect()

    for i in range(1):

        dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 230, 0, -65, 0, isQueued = 1)
        dType.SetWAITCmd(api, 200, 1)

        dType.SetJOGCmd(api, 1, 1, 1)
        dType.SetWAITCmd(api, 400, 1)
        dType.SetJOGCmd(api, 1, 0, 1)
        dType.SetWAITCmd(api, 600, 1)

        dType.SetJOGCmd(api, 1, 2, 1)
        dType.SetWAITCmd(api, 400, 1)
        dType.SetJOGCmd(api, 1, 0, 1)
        dType.SetWAITCmd(api, 200, 1)

def camera(conn):

    cap = cv2.VideoCapture(0)

    ball_x = []
    ball_y = []

    time_list = []

    x_v = []
    y_v = []
    V = []

    while(1):    

        _,frame = cap.read()
        box_points = findArucoMarkers(frame)

        print(len(box_points))

        if len(box_points) > 2:
            break

    while(1):

        _,frame = cap.read()
        cb_th = segmentaition(frame)
        px,py = find_ball(frame,cb_th,box_points)

        if px != None and py != None:

            ball_x.append(px)
            ball_y.append(py)

            t = time.time()

            time_list.append(t)
                
        elif px == None and len(ball_x) > 0:

            for i in range(len(ball_x)-1):

                xv = (ball_x[i+1] - ball_x[i])/(time_list[i+1] - time_list[i])
                yv = (ball_y[i+1] - ball_y[i])/(time_list[i+1] - time_list[i])

                vel = (math.sqrt(math.pow(ball_x[i+1] - ball_x[i], 2) + math.pow(ball_y[i+1] - ball_y[i], 2)))/(time_list[i+1] - time_list[i])

                x_v.append(xv)
                y_v.append(yv)
                V.append(vel)


            m_v = (math.sqrt(math.pow(ball_x[0] - ball_x[-1], 2) + math.pow(ball_y[0] - ball_y[-1], 2)))/(time_list[-1] - time_list[0])

            conn.send([x_v,y_v,V,time_list,m_v])

            break

        cv2.imshow('cam',frame)
        
        if cv2.waitKey(1) == 27:
            break

            

if __name__ == '__main__':

    dobot_connect()

    p_conn,c_conn = Pipe()

    p1 = Process(target=camera, args=(c_conn,))
    p1.start()

    hit()

    while(1):
        
        q = p_conn.recv()
        if len(q) > 1:
            x_v = q[0]
            y_v = q[1]
            V = q[2]
            time_list = q[3]
            m_v = q[4]

            time_list.pop()
        
            plt.subplot(3,1,1)
            plt.plot(time_list, x_v,'.-')
            plt.subplot(3,1,2)
            plt.plot(time_list,y_v,'.-')
            plt.subplot(3,1,3)
            plt.plot(time_list,V,'.-')

            print("m_v: %f cm/s" %(m_v))

            plt.show()

            break
            
