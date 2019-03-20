import cv2
import numpy as np
import math

def getConfig(path):
    configFile = open(path)
    configText = configFile.read()
    configFile.close()
    config = eval(configText)
    return config

# This method and the one below were copied from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x * 180 / math.pi, y * 180 / math.pi, z * 180 / math.pi]) # I modified the return statement to return degrees instead of radians

def getAngle(homography):
        print('homography: ' + str(homography))
        print('0th column: ' + str(homography[:,0]))
        print('1st column: ' + str(homography[:,1]))
        pose = np.eye(3, 4) #3x4 matrix, possibly not right size
        norm1 = np.linalg.norm(homography[:,0])
        norm2 = np.linalg.norm(homography[:,1])
        tnorm = (norm1+norm2) / 2

        v1 = np.array(homography[:,0])
        v2 = np.array(pose[:,0])
        cv2.normalize(v1, v2)

        pose[:,0] = v2

        cv2.normalize(homography[:,1], pose[:,1])

        v3 = np.cross(pose[:,0], pose[:,1])
        np.copyto(pose[2,2], v3)
        pose[3,3] = homography[2,2] / tnorm
        print(pose[3,3])

        return pose

