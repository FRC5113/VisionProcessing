import numpy as np
import cv2
from networktables import NetworkTables as nettab
import math
import threading
import FilterTests as ft
import Utils as utils
import os
import copy
import time

# The general loop of this file is explained here: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

# this gets the absolute path to the directory in which this file is located.
path = os.path.realpath(__file__)
path = path[:len(path) - path[::-1].index('/') - 1]

# This config has many settings that make it easy to flip between modes.
CONFIG = utils.getConfig(path + '/vision_processing.cfg') # This allows us to define constants from a configuration file,
                                                                                                 #rather than in the code
# defines our video source (number means camera, string means file path to video)
cap = cv2.VideoCapture(CONFIG['videoSource'])
if type(CONFIG['videoSource']) is int: # only configure the capture source if it is a camera, and it is only a camera if the source is an integer.
    os.system('sh ' + path + '/cameraconfig.sh') # This sets certain parameters for the camera such as disabling autoexposure, setting exposure, etc.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # idk if this really helps, but I think it helps with skipping frames. If the program were to process every frame, and the framerate were too slow, then the buffer would build up and there would be massive delay.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 426) # lower resolution leads to significantly higher framerate, which is more important.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Constants
RESOLUTION = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
LOWER_THRESH = np.array(CONFIG['LOWER_THRESH'])
UPPER_THRESH = np.array(CONFIG['UPPER_THRESH'])
ANGLE_CALIBRATION = CONFIG['ANGLE_CALIBRATION']
# Contour Test Constants
Y_TEST_THRESH = CONFIG['Y_TEST_THRESH']
MIN_SOLIDITY = CONFIG['MIN_SOLIDITY']
MIN_AREA = CONFIG['MIN_AREA']
HOMOGRAPHY_TEST_TOLERANCE = CONFIG['HOMOGRAPHY_TEST_TOLERANCE']

paused = False # press space to pause

########### Connects RaspberryPi to roboRIO ##############################
# copied from here: https://robotpy.readthedocs.io/en/stable/guide/nt.html#client-initialization-driver-station-coprocessor
def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

if not CONFIG['OFFLINE']:
    cond = threading.Condition()
    notified = [False]

    nettab.initialize(server='10.51.13.2') # ip of the roboRIO
    nettab.addConnectionListener(connectionListener, immediateNotify=True)

    table = nettab.getTable('contoursReport')
    with cond:
        print("Waiting")
        if not notified[0]:
            cond.wait()

        table.putNumber('X_RESOLUTION', RESOLUTION[0])
##########################################################################

# this is used to calculate frame rate (fps)
prevTime = time.time()

while True:
    ### Save new frame ###
    if not paused:
        ret, frame = cap.read()
        unmodified = copy.copy(frame) # just in case we want to display the raw video feed

    ############## Process Image #########################################
    ### Threshold image ###
    thresholdedImage = cv2.inRange(frame, LOWER_THRESH, UPPER_THRESH)

    ### Dilate image ###
    kernel = np.ones((5, 5), np.uint8) # This is a line copied from OpenCV's website. It establishes the size of the dilation
    dilatedImage = cv2.dilate(thresholdedImage, kernel, iterations = 1) # change number of iterations to dilate fupallerther
    ######################################################################

    ############## Get & Filter Contours #################################
    contoursImage, rawContours, hierarchy = cv2.findContours(dilatedImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''
    rawContours is a list of contours in the binary image (thresholdedImage and dilatedImage are binary images)
    A contour in a binary image is a border around where the white meets the black. We can now filter these
    to remove everything that also fit our threshold, so we are left only with the actual vision target.
    '''

    filteredContours = []

    for contour in rawContours:
        # if you want to know what these functions do, just look them up, they are fairly simple
        rectX, rectY, rectWidth, rectHeight = cv2.boundingRect(contour)
        contourArea = cv2.contourArea(contour)
        convexHull = cv2.convexHull(contour)

        # The "or not CONFIG['test']" line allows us to override the boolean in the config file
        yLevelTest = not CONFIG['y-levelTest'] or ft.ylevelTest(rectY + rectHeight / 2, Y_TEST_THRESH)
        solidityTest =  not CONFIG['solidityTest']or ft.solidityTest(contourArea, convexHull, MIN_SOLIDITY)
        quadrilateralTest = not CONFIG['quadrilateralTest'] or ft.quadrilateralTest(contour, 13)
        areaTest = not CONFIG['areaTest'] or ft.areaTest(contourArea, MIN_AREA)

        if yLevelTest and solidityTest and quadrilateralTest and areaTest:
            filteredContours.append(contour)
    ######################################################################

    ############## Determine Targets #########################################
    targets = []

    if len(filteredContours) < 2: # if we see less than two contours, we don't see the target.
        if not CONFIG['OFFLINE']:
            table.putBoolean('targetDetected', False)
    else:
        # Determine vision target:
        # these nested for loops find every combination of targets
        for i in range(len(filteredContours[:-1])): # [:-1] means all elements except for the last one
            contour = filteredContours[i]
            for j in range(i+1, len(filteredContours)):
                '''
                This portion finds four points of the vision target, and uses them to
                warp the perspective so that the image looks as if the camera were
                directly in front of the target. This allows us to make sure that
                we are really looking at a vision target, and not some random contour.
                '''
                
                contour2 = filteredContours[j]

                rect1 = cv2.minAreaRect(contour)
                rect2 = cv2.minAreaRect(contour2)
                rect1, rect2 = np.asarray(cv2.boxPoints(rect1)), np.asarray(cv2.boxPoints(rect2))
                
                # determines which target is on the left and which one is on the right
                if rect1[0][0] < rect2[0][0]: # checks x value of 0th point. Because the targets overlap, any point could be checked
                    leftTarget = rect1
                    rightTarget = rect2
                else:
                    leftTarget = rect2
                    rightTarget = rect1

                ############### correctly sorts the points of the contours into the following order ##################
                # [topLeft, topRight, bottomRight, bottomLeft]
                leftTargetVertices = [[leftTarget[0][0], leftTarget[0][1]]] * 4 # an array consisting of the four elements, all of which are the first corner of the left target
                rightTargetVertices = [[rightTarget[0][0], rightTarget[0][1]]] * 4 # an array consisting of the four elements, all of which are the first corner of the left target
    
                for point in leftTarget:
                    point = [point[0], point[1]] # point is formatted as an np array, so this formats it as a normal array.
                    if point[1] < leftTargetVertices[0][1]:
                        leftTargetVertices[0] = point
                    if point[0] > leftTargetVertices[1][0]:
                        leftTargetVertices[1] = point
                    if point[1] > leftTargetVertices[2][1]:
                        leftTargetVertices[2] = point
                    if point[0] < leftTargetVertices[3][0]:
                        leftTargetVertices[3] = point
    
                for point in rightTarget:
                    point = [point[0], point[1]]
                    if point[0] < rightTargetVertices[0][0]:
                        rightTargetVertices[0] = point
                    if point[1] < rightTargetVertices[1][1]:
                        rightTargetVertices[1] = point
                    if point[0] > rightTargetVertices[2][0]:
                        rightTargetVertices[2] = point
                    if point[1] > rightTargetVertices[3][1]:
                        rightTargetVertices[3] = point
                ############################################################################################################
    
                # I only use four points for srcPoints and destPoints because I have 8 total, and I need to see if the other four are in the correct position.
                # If I used all 8, then they all would, by definition, be in the correct spot, because that's what findHomography does.
                srcPoints = np.asarray([leftTargetVertices[0], rightTargetVertices[1], rightTargetVertices[2], leftTargetVertices[3]])
                destPoints = np.asarray([
                        CONFIG['HOMOGRAPHY_TEST_POINTS'][0],
                        CONFIG['HOMOGRAPHY_TEST_POINTS'][5],
                        CONFIG['HOMOGRAPHY_TEST_POINTS'][6],
                        CONFIG['HOMOGRAPHY_TEST_POINTS'][3]
                ])

                CONFIG['DEST_POINTS'] = destPoints
    
                h, mask = cv2.findHomography(srcPoints, destPoints, cv2.RANSAC, 5)
                
                #warped = cv2.warpPerspective(frame, h, (int(RESOLUTION[0]), int(RESOLUTION[1]))) # this corrects the perspective of the entire frame
                warpedPoints = cv2.perspectiveTransform(np.array([leftTargetVertices, rightTargetVertices], dtype='float32'), h) # this corrects the perspective of just the given vertices
    
                warpedLeft = warpedPoints[0].astype(int)
                warpedRight = warpedPoints[1].astype(int)
    
                #ctr = np.array(srcPoints).reshape((-1,1,2)).astype(np.int32) # this line is copied from https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python
                #cv2.drawContours(frame, [ctr], -1, (255, 0, 0), 2)
                #cv2.drawContours(warped, [warpedLeft, warpedRight], -1, (255, 0, 0), 2)
                #cv2.imshow('warped', warped)
    
                # 'superStrictTest' returns true if every vertex is within a certain threshold of where a vision target's vertex would be
                if ft.superStrictTest(warpedLeft, warpedRight, CONFIG['HOMOGRAPHY_TEST_POINTS'], HOMOGRAPHY_TEST_TOLERANCE):
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    x2, y2, w2, h2 = cv2.boundingRect(contour2)
                    centerX = (x1 + w1 / 2 + x2 + w2 / 2) / 2

                    #theta = 90 - math.atan2(h[1,1], h[1,0]) * 180 / math.pi # copied from answers.opencv.org/question/203890/how-to-find-rotation-angle-from-homography-matrix/

                    '''
                    essentialMat, mask2 = cv2.findEssentialMat(np.array(leftTargetVertices + rightTargetVertices), np.array(CONFIG['HOMOGRAPHY_TEST_POINTS']))
                    pose = cv2.recoverPose(essentialMat, srcPoints, destPoints)
                    recoveredRotation = pose[1]
                    print(utils.rotationMatrixToEulerAngles(recoveredRotation)[2])
                    print(pose[2][1])
                    print('aaa')
                    print('bbb')
                    retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(recoveredRotation)
                    print('retval: ' + str(retval))
                    print('mtxR: ' + str(mtxR))
                    print('mtxQ: ' + str(mtxQ))
                    print('Qx: ' + str(Qx))
                    print('Qy: ' + str(Qy))
                    print('Qz: ' + str(Qz))
                    '''

                    objectPoints = CONFIG['REAL_WORLD_POINTS']
                    imagePoints = leftTargetVertices + rightTargetVertices #CONFIG['HOMOGRAPHY_TEST_POINTS']
                    print(cv2.calibrateCamera(np.array([objectPoints], dtype=np.float32), np.array([imagePoints], dtype=np.float32), RESOLUTION, None, None)[1]) # camera matrix
                    #cameraMatrix = CONFIG['CAMERA_MATRIX']
                    #rotationVector = cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs)[1]
                    #rotationMatrix = cv2.Rodrigues(rotationVector)[0]
                    #print(rotationMatrix)

                    #targets.append({'contours': [contour, contour2], 'x': centerX, 'angle': theta})
                    targets.append({'contours': [contour, contour2], 'x': centerX, 'angle': 0})

                    #print(theta)
    
                cv2.drawContours(frame, filteredContours, -1, (0, 0, 255), 2)
    
                print('-----------------------------------------------')
    
    # Now we have our targets!
    # Let's select the target closest to the center. We may change how we select targets in the future

    if len(targets) > 0:
        selectedTarget = targets[0]
        
        for target in targets:
            if math.fabs(target['x'] - RESOLUTION[0] / 2) < math.fabs(selectedTarget['x'] - RESOLUTION[0]):
                selectedTarget = target
    else:
        selectedTarget = None

    if selectedTarget != None:
        cv2.drawContours(frame, selectedTarget['contours'], -1, (0, 255, 0), 2)
    ######################################################################

    ########################## Send Data #################################
    if not CONFIG['OFFLINE']:
        if selectedTarget != None:
            print('target found')
            table.putBoolean('targetDetected', True)
            table.putNumber('xCoord', selectedTarget['x'])
            table.putNumber('angle', selectedTarget['angle'])
        else:
            print('target not found')
            table.putBoolean('targetDetected', False)
    ######################################################################

    ############## Show All Images #######################################
    if CONFIG['DISPLAY']:
        cv2.rectangle(frame, (0, int(Y_TEST_THRESH[0] - Y_TEST_THRESH[1] / 2)), (int(RESOLUTION[0]), Y_TEST_THRESH[0] + Y_TEST_THRESH[1]), (0, 255, 255), 2)

        #cv2.imshow('thresholded', thresholdedImage)
        #cv2.imshow('dilated', dilatedImage)
        cv2.imshow('contours', frame)
        #cv2.imshow('unmodified', unmodified)
    ######################################################################

    timeLength = time.time() - prevTime
    print('{} fps'.format(int(1 / timeLength))) # one frame divided by # of seconds the frame took to process = frames per second
    prevTime = time.time()
    print(ANGLE_CALIBRATION)

    print('========================================================')

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('c'): # c for callibration. These values are then copied into the config file.
        if selectedTarget != None:
            ANGLE_CALIBRATION += selectedTarget['angle']
            print("    'ANGLE_CALIBRATION': " + str(ANGLE_CALIBRATION))
        print("    'HOMOGRAPHY_TEST_POINTS': [")
        print('         [{}, {}], # top left vertex of leftTarget'.format(int(leftTargetVertices[0][0]), int(leftTargetVertices[0][1])))
        print('         [{}, {}], # top right vertex of leftTarget'.format(int(leftTargetVertices[1][0]), int(leftTargetVertices[1][1])))
        print('         [{}, {}], # bottom right vertex of leftTarget'.format(int(leftTargetVertices[2][0]), int(leftTargetVertices[2][1])))
        print('         [{}, {}], # bottom left vertex of leftTarget'.format(int(leftTargetVertices[3][0]), int(leftTargetVertices[3][1])))
        print()
        print('         [{}, {}], # top left vertex of rightTarget'.format(int(rightTargetVertices[0][0]), int(rightTargetVertices[0][1])))
        print('         [{}, {}], # top right vertex of rightTarget'.format(int(rightTargetVertices[1][0]), int(rightTargetVertices[1][1])))
        print('         [{}, {}], # bottom right vertex of rightTarget'.format(int(rightTargetVertices[2][0]), int(rightTargetVertices[2][1])))
        print('         [{}, {}]],# bottom left vertex of rightTarget'.format(int(rightTargetVertices[3][0]), int(rightTargetVertices[3][1])))
    elif key & 0xFF == ord('l'): # l for latency test
        table.putBoolean('latency test', True)
    elif key & 0xFF == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
