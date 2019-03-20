import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(sys.argv[1])
ret, frame = cap.read()
cv2.imshow('frame',frame)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('cropped.avi.tmp', fourcc, 20.0, (640,480))

while(cap.isOpened()):
    if ret==True:
        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif key == ord(' '):
            ret, frame = cap.read()
            cv2.imshow('frame',frame)
        elif key == ord('r'):
            out.write(frame)
            ret, frame = cap.read()
            cv2.imshow('frame',frame)

    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

