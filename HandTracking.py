import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    isSuccess, videoFrame = cap.read()
    imgRGB = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand.landmark):
                height, width, channels = videoFrame.shape
                center_x, center_y = int(landmark.x * width), int(
                    landmark.y * height
                )  # as landmark.x, landmark.y will not provide values in pixels
                print(id, center_x, center_y)
                if id == 4:
                    cv2.circle(
                        videoFrame,
                        (center_x, center_y),
                        15,
                        (255, 0, 255),
                        cv2.FILLED,
                    )
            mpDraw.draw_landmarks(videoFrame, hand, mpHands.HAND_CONNECTIONS)

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
