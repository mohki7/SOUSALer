from flask import Flask, render_template, Response
import cv2
from camera import VideoCamera
import HandTrackingModule as htm
import time
import pyautogui

app = Flask(__name__)
camera = cv2.VideoCapture(0)
# 画面サイズ
size = (600, 400)

# ビデオストリーミングのジェネレータ関数


def gen_frames(camera):
    while True:
        success, frame = camera.read()

        if not success:
            print('cannot load the camera')
            break
        else:
            # MJPEGストリームに必要なマルチパートメッセージを生成
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# SSEのジェネレータ関数
# HTMLに指の本数を渡して表示させる


def gen_finger_count(camera):
    finger_counts = []
    last_count = None
    while True:
        success, frame, totalFingers = camera.get_frame()
        if not success:
            print('cannot load the camera')
            break
        else:
            finger_counts.append(totalFingers)
            # リストの最後の二つの要素が同じかどうかを確認
            if len(finger_counts) > 1 and finger_counts[-1] == finger_counts[-2]:
                last_count = finger_counts[-1]
            else:
                last_count = None
                finger_counts = []
            # SSEを使用して指の数をHTMLページに送信する
            # 2秒以上同じ数字が続いた場合はその数字を表示する
            if last_count is not None and len(finger_counts) >= 20:
                yield f"data {last_count}\n\n".encode()
            else:
                yield f"data: {totalFingers}\n\n".encode()
            time.sleep(0.1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    # return Response(gen_frames(VideoCamera(), mimetype='multipart/x-mixed-replace; boundary=frame'))
    return Response(gen_frames(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/finger_count')
# def finger_count():
#     return Response(gen_finger_count(VideoCamera()), mimetype='text/event-stream')


@app.route('/finger_count')
def finger_count():
    for finger_count in gen_finger_count(VideoCamera()):
        print(finger_count.strip())
        if finger_count.strip() == b'data: 1':
            # 手のパターンが1だったら拡大する
            print('Zoom In')
            pyautogui.keyDown('command')
            pyautogui.press('+')
            pyautogui.keyUp('command')
        elif finger_count.strip() == b'data: 2':
            # 手のパターンが2だったらピンチアウトする
            print('Zoom Out')
            pyautogui.keyDown('command')
            pyautogui.press('-')
            pyautogui.keyUp('command')
        elif finger_count.strip() == b'data: 3':
            print('screenshot!')
            pyautogui.screenshot('screenshot.png')

        elif finger_count.strip() == b'data: 4':
            # 現在のマウスカーソル位置を取得
            m_posi_x, m_posi_y = pyautogui.position()

            # スクロール
            pyautogui.scroll(-2000, m_posi_x, m_posi_y)
            print('scroll!')
            # pyautogui.screenshot('screenshot.png')
        elif finger_count.strip() == b'data: 5':
            print('play!')
            pyautogui.press(' ')
        # yield finger_count


if __name__ == '__main__':
    app.run(debug=True)
