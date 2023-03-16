import cv2
import mediapipe as mp
import time
# Used to convert protobuf message
# to a dictionary.
from google.protobuf.json_format import MessageToDict


class handDetector():
    def __init__(self, mode=True, maxHands=2, detectionCon=0, trackCon=0.5):
        """
        mode:手の検出モード。Falseは単一のみ。Trueは複数
        maxHands:検出する手の最大数
        detectionCon:手を検出するための最低信頼度。動画では0.5だったが、エラー。0だと動く
        trackCon:手をトラックするための最低信頼度
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # 手を検知する設定。なんかこの.Hands()の使い方解説してた
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        手を見つける。手の関節の位置に印をつけたimgと、その手がどっちの手なのかのwhichhandsを返す
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 手の検出。手の位置や特徴点などをresultsに格納。手がない時はNone。hands()は動画内で説明あり
        self.results = self.hands.process(imgRGB)
        whichhands = ''

        if self.results.multi_hand_landmarks:  # 画像に手が検出できた時。
            for handLms in self.results.multi_hand_landmarks:
                if draw:  # 手の指の位置に円を描画。draw=Falseで描画しない
                    # HAND_CONNECTIONSで点を結ぶ線を描画-
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

            # 右手か左手かを返す。なぜか反転されて出てくる
            for i in self.results.multi_handedness:
                # Return whether it is Right or Left Hand
                whichhands = MessageToDict(i)['classification'][0]['label']

        return img, whichhands

    def findPosition(self, img, handNo=0, draw=True):
        """
        検出した手の指の位置をリストで返す
        """
        lmList = []
        if self.results.multi_hand_landmarks:  # 手を検出できた時
            myHand = self.results.multi_hand_landmarks[handNo]  # 最初に検出された手を代入

            for id, lm in enumerate(myHand.landmark):
                # 位置を計算
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # それぞれの点の場所と位置を追加
                lmList.append([id, cx, cy])
                if draw:  # draw=Falseで円を表示しないことも可能
                    # imgの中、(cx, cy)の場所に大きさ10で色を(255, 0, 255)で表示。cv2.FILLEDは塗りつぶし？
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # 0はカメラのid
    detector = handDetector()

    while True:
        success, img = cap.read()  # カメラからの入力を受け取る
        img = detector.findHands(img)  # 手の検知。
        lmList = detector.findPosition(img)  # 場所の計算
        if len(lmList) != 0:  # 手を検知できた時
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
