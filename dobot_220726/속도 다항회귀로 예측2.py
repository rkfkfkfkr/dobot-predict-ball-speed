import numpy as np
import numpy.random as rnd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import cv2
import imutils
import threading
import math
import time
import DobotDllType as dType
from numpy.linalg import inv

import cv2.aruco as aruco
import os

from multiprocessing import Process, Pipe, Queue, Value, Array, Lock

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

def main():

    cap = cv2.VideoCapture(0)

    ball_x = []
    ball_y = []

    ball_v = []

    time_list = []

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
            
        elif px == None and py == None:

            for i in range(len(ball_x)-1):

                vel = math.sqrt((math.pow(ball_x[i+1] - ball_x[i],2) + math.pow(ball_y[i+1] - ball_y[i],2)))/(time_list[i+1] - time_list[i])
                ball_v.append(vel)

            if len(ball_v) > 0:

                #pt = time_list[-1]
                
                #time_list.pop()
                ball_y.pop()

                X = np.reshape(ball_y,(-1,1))
                y = np.reshape(ball_v,(-1,1))

                poly_features = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly_features.fit_transform(X)

                lin_reg = LinearRegression()
                lin_reg.fit(X_poly, y)
                lin_reg.intercept_, lin_reg.coef_
                X_new=np.linspace(float(ball_y[0]), float(ball_y[-1]), 100).reshape(100, 1)

                X_new_poly = poly_features.transform(X_new)
                y_new = lin_reg.predict(X_new_poly)

                plt.plot(X, y, ".-")
                plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
                plt.xlabel("$time$", fontsize=14)
                plt.ylabel("$Vel$", rotation=0, fontsize=14)
                plt.legend(loc="upper left", fontsize=12)

                py = [120, 120**2]
                py = np.reshape(py,(-1,1))
                pv = lin_reg.predict(py)

                print(pv)

                print()
                
                break

         
        cv2.imshow('cam',frame)
        
        if cv2.waitKey(1) == 27:
            break

    plt.show()
    cap.release()
    cv2.destroyAllWindows()

main()
