import cv2
import mediapipe as mp
import PySimpleGUI as sg
import numpy as np

# MediaPipe初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# PySimpleGUIの設定
layout = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Exit()]
]

window = sg.Window('Hand Tracking Trajectory', layout, resizable=True, finalize=True)
canvas_elem = window['-IMAGE-']
canvas = canvas_elem.Widget

# カメラ初期化
cap = cv2.VideoCapture(0)

# 手の検出モデルの初期化
with mp_hands.Hands(max_num_hands=1) as hands:
    trajectory_points = []  # 指の動きの軌跡を記録するリスト

    while True:
        event, values = window.read(timeout=20)

        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break

        ret, frame = cap.read()
        if not ret:
            continue

        # フレームをBGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手の検出
        results = hands.process(rgb_frame)
        
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 手の位置を取得
                h, w, _ = frame.shape
                cx, cy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)

                # 軌跡に座標を追加
                trajectory_points.append((cx, cy))

                # 軌跡を描画
                if len(trajectory_points) > 1:
                    cv2.polylines(frame, [np.array(trajectory_points, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=2)

        # OpenCV画像をPySimpleGUIに反映
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        canvas_elem.update(data=imgbytes)

    window.close()

# 後片付け
cap.release()
cv2.destroyAllWindows()
