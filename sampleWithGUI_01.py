# ステップ1. インポート
import PySimpleGUI as sg  # PySimpleGUIをsgという名前でインポート
import os  # OS依存の操作（Pathやフォルダ操作など）用ライブラリのインポート
import numpy as np  # numpyのインポート
import cv2  # OpenCV（python版）のインポート
import mediapipe as mp  # mediapipeのインクルード
import time

# 画像リサイズ関数（高さが指定した値になるようにリサイズ (アスペクト比を固定)）
def scale_to_height(img, height):
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))

    return dst
BAR_MAX = 1000

# ---- 顔認識エンジンセット ----
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ---- 大域変数 ----
display_size = (400, 300)  # ディスプレイサイズ
isOpened = 0  # カメラがオープンになっているかどうかのフラグ
isRun = 0  # 認識On/Offフラグ（Onの時に認識実行）

# ステップ2. デザインテーマの設定
sg.theme("Default1")



# ステップ3. ウィンドウの部品とレイアウト
column_to_be_centered = [
    [sg.Text('Media Pipe', size=(40, 1), justification='center', font='Helvetica 20')],
    [sg.Image(filename="", key="-input_image-")],
    [sg.Button('Record', key="camera", size=(10, 1), font='Helvetica 14'),
    sg.Button('Save', key="save", size=(10, 1), font='Any 14'),
    sg.Button('Exit', key="exit", size=(10, 1), font='Helvetica 14'), ],
    [
        sg.Text("ランドマークの表示", size=(15, 1)),
        sg.Combo(("ON", "OFF"), default_value="ON", size=(5, 1), key="landmark"),
    ],
]

layout = [
    [sg.VPush()],
    [sg.Push(), sg.Column(column_to_be_centered,element_justification='c'), sg.Push()],
    [sg.VPush()],
]


# ステップ4. ウィンドウの生成
window = sg.Window("人体を認識するツール", layout, location=(400, 20))

# ステップ5. カメラ，mediapipeの初期設定
cap = None  # カメラを初期化

with mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    out = None  # VideoWriterを初期化

    # ステップ6. イベントループ
    while True:
        event, values = window.read(timeout=10)

        if event in (None, "exit"):  # ウィンドウのXボタンまたは”終了”ボタンを押したときの処理
            break

        if event == "camera":  # 「カメラ」ボタンが押された時の処理
            if cap is None:
                print("Camera Open")
                cap = cv2.VideoCapture(0)  # 任意のカメラ番号に変更する
                isOpened, orig_img = cap.read()
                if isOpened:  # 正常にフレームを読み込めたら
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                    out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))
                    print("Frame size = " + str(orig_img.shape))
                    print("fps=",cap.get(cv2.CAP_PROP_FPS))
                    # 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
                    disp_img = scale_to_height(orig_img, display_size[1])
                    # 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
                    imgbytes = cv2.imencode(".png", disp_img)[1].tobytes()
                    # ウィンドウへ表示
                    window["-input_image-"].update(data=imgbytes)
                else:
                    print("Cannot capture a frame image")
        elif event == "save":  # 「保存」ボタンが押されたときの処理
            if out is not None:
                out.release()  # VideoWriterを解放
                print("Write movie -> out.mp4")
                out = None  # VideoWriterを初期化
            else:
                print("Start saving")
        elif isOpened == 1:
            # ---- フレーム読み込み ----
            ret, frame = cap.read()
            if ret:  # 正常にフレームを読み込めたら
                if out is not None:
                    # VideoWriterでムービー書き出し処理を書く
                    time.sleep(1 / fps)
                    out.write(frame)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
                results = holistic.process(image)  # 検出
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGBからBGRに変換

                if values["landmark"] == "ON":
                    # 顔
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(
                            color=(255, 0, 0), thickness=2, circle_radius=1
                        ),
                        mp_drawing.DrawingSpec(
                            color=(255, 255, 255), thickness=2, circle_radius=1
                        ),
                    )

                # 右手
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=1
                    ),
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=2, circle_radius=1
                    ),
                )

                # 左手
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=1
                    ),
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=2, circle_radius=1
                    ),
                )

                # 姿勢
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=1
                    ),
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=2, circle_radius=1
                    ),
                )
                # 表示用に画像を固定サイズに変更
                disp_img = scale_to_height(image, display_size[1])
                # 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
                imgbytes = cv2.imencode(".png", disp_img)[1].tobytes()
                # ウィンドウへ表示
                window["-input_image-"].update(data=imgbytes)

# カメラとウィンドウを解放
if cap is not None:
    cap.release()
if out is not None:
    out.release()
window.close()

# https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_OpenCV_Webcam.py