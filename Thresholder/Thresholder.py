import cv2
import numpy as np
import sys
import copy

################ Setting up VideoCapture ##################
videoSource = sys.argv[1]

if videoSource.isdigit():
    videoSource = int(videoSource)

cap = cv2.VideoCapture(videoSource)
###########################################################

configFile = open('thresholder.cfg')
#CONFIG = eval(configFile.read())
configFile.close()

LOWER_THRESH = np.array([0, 0, 0])
UPPER_THRESH = np.array([360, 255, 255])
paused = False

def nothing(x): # defines an empty function. cv2.createTrackbar() requires
    pass        # function, and since we don't use one, we just defined an

cv2.namedWindow('threshold_tester')
cv2.createTrackbar('h', 'threshold_tester', LOWER_THRESH[0], 360, nothing)
cv2.createTrackbar('s', 'threshold_tester', LOWER_THRESH[1], 255, nothing)
cv2.createTrackbar('v', 'threshold_tester', LOWER_THRESH[2], 255, nothing)
cv2.createTrackbar('H', 'threshold_tester', UPPER_THRESH[0], 360, nothing)
cv2.createTrackbar('S', 'threshold_tester', UPPER_THRESH[1], 255, nothing)
cv2.createTrackbar('V', 'threshold_tester', UPPER_THRESH[2], 255, nothing)

while cap.isOpened():
    if cv2.getTrackbarPos('h', 'threshold_tester') > UPPER_THRESH[0]:
        cv2.setTrackbarPos('h', 'threshold_tester', UPPER_THRESH[0])
    if cv2.getTrackbarPos('s', 'threshold_tester') > UPPER_THRESH[1]:
        cv2.setTrackbarPos('s', 'threshold_tester', UPPER_THRESH[1])
    if cv2.getTrackbarPos('v', 'threshold_tester') > UPPER_THRESH[2]:
        cv2.setTrackbarPos('v', 'threshold_tester', UPPER_THRESH[2])
    if cv2.getTrackbarPos('H', 'threshold_tester') < LOWER_THRESH[0]:
        cv2.setTrackbarPos('H', 'threshold_tester', LOWER_THRESH[0])
    if cv2.getTrackbarPos('S', 'threshold_tester') < LOWER_THRESH[1]:
        cv2.setTrackbarPos('S', 'threshold_tester', LOWER_THRESH[1])
    if cv2.getTrackbarPos('V', 'threshold_tester') < LOWER_THRESH[2]:
        cv2.setTrackbarPos('V', 'threshold_tester', LOWER_THRESH[2])

    LOWER_THRESH[0] = cv2.getTrackbarPos('h', 'threshold_tester')
    LOWER_THRESH[1] = cv2.getTrackbarPos('s', 'threshold_tester')
    LOWER_THRESH[2] = cv2.getTrackbarPos('v', 'threshold_tester')
    UPPER_THRESH[0] = cv2.getTrackbarPos('H', 'threshold_tester')
    UPPER_THRESH[1] = cv2.getTrackbarPos('S', 'threshold_tester')
    UPPER_THRESH[2] = cv2.getTrackbarPos('V', 'threshold_tester')

    if not paused:
        ret, frame = cap.read()

    try:
        resolution = frame.shape
    except:
        break
    thresholdedImage = cv2.inRange(frame, LOWER_THRESH, UPPER_THRESH)
    thresholdedImage = cv2.cvtColor(thresholdedImage, cv2.COLOR_GRAY2BGR)
    finalThreshold = copy.copy(thresholdedImage)
    solidColor = np.zeros((resolution[0], resolution[1], 3), np.uint8) # black

    lowHue = cv2.inRange(frame, np.array([0, 0, 0]), np.array([LOWER_THRESH[0], 255, 255]))
    solidColor[:] = (254, 0, 0) # blue
    lowHue = cv2.bitwise_and(solidColor, solidColor, mask=lowHue)
    thresholdedImage = cv2.add(thresholdedImage, lowHue)

    lowSat = cv2.inRange(frame, np.array([0, 0, 0]), np.array([360, LOWER_THRESH[1], 255]))
    solidColor[:] = (0, 254, 0) # green
    lowSat = cv2.bitwise_and(solidColor, solidColor, mask=lowSat)
    thresholdedImage = cv2.add(thresholdedImage, lowSat)

    lowVal = cv2.inRange(frame, np.array([0, 0, 0]), np.array([360, 255, LOWER_THRESH[2]]))
    solidColor[:] = (0, 0, 254) # red
    lowVal = cv2.bitwise_and(solidColor, solidColor, mask=lowVal)
    thresholdedImage = cv2.add(thresholdedImage, lowVal)

    highHue = cv2.inRange(frame, np.array([UPPER_THRESH[0], 0, 0]), np.array([360, 255, 255]))
    solidColor[:] = (254, 0, 0) # blue
    highHue = cv2.bitwise_and(solidColor, solidColor, mask=highHue)
    thresholdedImage = cv2.add(thresholdedImage, highHue)

    highSat = cv2.inRange(frame, np.array([0, UPPER_THRESH[1], 0]), np.array([360, 255, 255]))
    solidColor[:] = (0, 254, 0) # green
    highSat = cv2.bitwise_and(solidColor, solidColor, mask=highSat)
    thresholdedImage = cv2.add(thresholdedImage, highSat)

    highVal = cv2.inRange(frame, np.array([0, 0, UPPER_THRESH[2]]), np.array([360, 255, 255]))
    solidColor[:] = (0, 0, 254) # red
    highVal = cv2.bitwise_and(solidColor, solidColor, mask=highVal)
    thresholdedImage = cv2.add(thresholdedImage, highVal)

    thresholdedImage[np.where((thresholdedImage == [254, 254, 254]).all(axis=2))] = [0, 0, 0]

    #cv2.imshow('frame', frame)
    cv2.imshow('thresholdedImage', thresholdedImage)
    cv2.imshow('finalThreshold', finalThreshold)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):
        paused = not paused
    elif key & 0xFF == ord('p'):
        print('LOWER_THRESH = [{}, {}, {}]'.format(LOWER_THRESH[0], LOWER_THRESH[1], LOWER_THRESH[2]))
        print('UPPER_THRESH = [{}, {}, {}]'.format(UPPER_THRESH[0], UPPER_THRESH[1], UPPER_THRESH[2]))
        print("LOWER_THRESH': [{}, {}, {}]".format(LOWER_THRESH[0], LOWER_THRESH[1], LOWER_THRESH[2]))
        print("UPPER_THRESH': [{}, {}, {}]".format(UPPER_THRESH[0], UPPER_THRESH[1], UPPER_THRESH[2]))

print('LOWER_THRESH = [{}, {}, {}]'.format(LOWER_THRESH[0], LOWER_THRESH[1], LOWER_THRESH[2]))
print('UPPER_THRESH = [{}, {}, {}]'.format(UPPER_THRESH[0], UPPER_THRESH[1], UPPER_THRESH[2]))
print("'LOWER_THRESH': [{}, {}, {}],".format(LOWER_THRESH[0], LOWER_THRESH[1], LOWER_THRESH[2]))
print("'UPPER_THRESH': [{}, {}, {}],".format(UPPER_THRESH[0], UPPER_THRESH[1], UPPER_THRESH[2]))

cap.release()
cv2.destroyAllWindows()
