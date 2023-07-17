import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCount=0.5, trackCount=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCount = detectionCount
        self.trackCount = trackCount

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, videoFrame, draw=True):
        imgRGB = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        videoFrame, hand, self.mpHands.HAND_CONNECTIONS
                    )
        return videoFrame

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landmark in enumerate(myHand.landmark):
                height, width, channels = img.shape
                center_x, center_y = int(landmark.x * width), int(
                    landmark.y * height
                )  # as landmark.x, landmark.y will not provide values in pixels
                # print(id, center_x, center_y)
                lmList.append([id, center_x, center_y])
                # if id == 4:
                if draw:
                    cv2.circle(
                        img,
                        (center_x, center_y),
                        15,
                        (255, 0, 255),
                        cv2.FILLED,
                    )
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        isSuccess, videoFrame = cap.read()
        videoFrame = detector.findHands(videoFrame)
        lmList = detector.findPosition(videoFrame)
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


if __name__ == "__main__":
    main()
