import cv2
import HandTrackingModule as htm
import time

detector = htm.handDetector(maxHands=1)  # hand detectionの中身

# ---手の画像の設定---
folderPath = './static/FingerImages'
myList = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png']
overlayList = []  # 写真を入れるリスト

# mediapipeの指先の数字　# https://google.github.io/mediapipe/solutions/hands
tipIds = [4, 8, 12, 16, 20]
# それぞれの指の先端
# 親指、人差し指、中指、薬指、小指の順

# ---手の画像の設定---


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.prev_fingers = 0
        self.prev_time = time.time()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # load video
        # success: 成功したかどうか
        # frame: フレーム情報
        success, frame = self.video.read()

        """
        hand detection algorithm↓
        """
        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            overlayList.append(image)  # 写真をoverlayListに格納

        img, whichhands = detector.findHands(
            frame)  # 画像frameの中から手を見つけ、imgに格納。imgは何者?
        # 手の場所の計算。[[指のランドマーク, x座標, y座標]]
        # [[0, 629, 1082], [1, 813, 1042], [2, 967, 942], [3, 1068, 847], [4, 1143, 773], [5, 854, 674], [6, 932, 516], [7, 982, 415], [8, 1022, 329], [9, 733, 632], [10, 772, 431], [11, 803, 306], [12, 829, 202], [13, 613, 640], [14, 636, 447], [15, 664, 325], [16, 694, 227], [17, 489, 686], [18, 472, 527], [19, 476, 422], [20, 492, 332]]
        lmList = detector.findPosition(img)
        totalFingers = 0
        if len(lmList) != 0:  # 手を検知できた時
            fingers = []
            # 右手の時
            if whichhands == "Left":  # なぜか返ってくる値が反転してる
                # thumb
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # 親指の先端のx座標 > 親指の第1関節のx座標の時
                    fingers.append(1)
                else:
                    fingers.append(0)
            # 左手の時
            elif whichhands == 'Right':
                # thumb
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:  # 親指の先端のx座標 < 親指の第1関節のx座標の時
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):  # 親指以外の処理
                # 指の関節の位置で指がどうなっているかを判断
                # 親指以外の指の先端のy座標 < 親指以外の指の第二関節のy座標
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # fingersの中で1になっている数を数えてtotalFingersに代入
            totalFingers = fingers.count(1)

            # 2秒以上同じ数字が検出された場合は、数字を更新してHTMLページに送信
            if totalFingers == self.prev_fingers and time.time() - self.prev_time >= 2:
                # print(time.time())
                # print(self.prev_time)
                print(totalFingers)
                self.prev_time = time.time()
                return True, img, totalFingers

            # 2秒以上同じ数字が検出されない場合は、前回の数字をHTMLページに送信
            elif totalFingers != self.prev_fingers:
                self.prev_fingers = totalFingers
                self.prev_time = time.time()
                # print(f'prev_fingers:{self.prev_fingers}')
                return True, img, None

            # h, w, c = overlayList[totalFingers - 1].shape

            # img[0:h, 0:w] = overlayList[totalFingers - 1]

            # cv2.rectangle(img, (20, 255), (170, 425),
            #               (0, 255, 0), cv2.FILLED)
            # cv2.putText(img, str(totalFingers), (45, 375),
            #             cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
        """
        hand detection algorithm↑
        """

        ret, buffer = cv2.imencode('.jpg', frame)

        # return success, buffer.tobytes(), totalFingers

        return success, buffer.tobytes(), None

    # read()は、二つの値を返すので、success, imageの2つ変数で受ける
    # OpencVはデフォルトでは raw imagesなので JPEGに変換
    # ファイルに保存する場合はimwriteを使用、メモリ上に格納したい時はimencodeを使用
    # cv2.imencode() は numpy.ndarray() を返すので .tobytes() で bytes 型に変換
