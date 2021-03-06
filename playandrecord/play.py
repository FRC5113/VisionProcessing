import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(sys.argv[1])

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    #print(cap.get(cv2.CAP_PROP_FPS))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
