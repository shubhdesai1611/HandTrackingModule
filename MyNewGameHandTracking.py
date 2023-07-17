import cv2
import mediapipe as mp
import time
import HandtrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)
detector = htm.handDetector()
while True:
    isSuccess, videoFrame = cap.read()
    videoFrame = detector.findHands(videoFrame, draw=True)
    lmList = detector.findPosition(videoFrame, draw=True)
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        videoFrame,
        str(int(fps)),
        (10, 70),
        cv2.FONT_HERSHEY_COMPLEX,
        3,
        (255, 0, 255),
        3,
    )

    cv2.imshow("My Frame", videoFrame)
    cv2.waitKey(1)
