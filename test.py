import cv2
import mediapipe as mp
import PySimpleGUI as sg
import numpy as np

# 画像リサイズ関数（高さが指定した値になるようにリサイズ (アスペクト比を固定)）
def scale_to_height(img, height):
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))
    return dst

display_size = (400, 300)  # ディスプレイサイズ
sg.theme("Default1")

# MediaPipe初期化
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# PySimpleGUIの設定
layout_main = [
    [sg.Image(filename='', key='-IMAGE-')],
    [sg.Button('サブ画面を表示', key="show_subWin", size=(10, 1)), 
     sg.Exit(key="exit", size=(10, 1))],
    [sg.Button('CLEAR', key="clear", size=(10, 1))],
]

window = sg.Window('Hand Tracking Trajectory', layout_main, resizable=True, finalize=True)
canvas_elem = window['-IMAGE-']
canvas = canvas_elem.Widget

layout_sub = [
    [sg.Text('sub')],
    [sg.Button('サブ画面を閉じる', key="close_subWin", size=(10, 1))]
]

window_sub = sg.Window('サブ画面', layout_sub, finalize=True)
window_sub.hide()  # 初めは非表示

# カメラ初期化
cap = cv2.VideoCapture(0)

# 手の検出モデルの初期化
with mp_hands.Hands(max_num_hands=1) as hands:
    trajectories = []  # 複数の軌跡を管理するリスト

    while True:
        event, values = window.read(timeout=20)
        event_sub, values_sub,  = window_sub.read(timeout=20)

        if event in (sg.WINDOW_CLOSED, "exit"):
            break
        elif event == "show_subWin":
            window_sub.un_hide()
        elif event_sub == "close_subWin":
            window_sub.hide()
        elif event == "clear":
            trajectories = []  # サブ画面を閉じると軌跡もリセット
            

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 画像反転

        if not ret:
            continue

        # フレームをBGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手の検出
        results = hands.process(rgb_frame)

        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # 最初に検出された手のみを使用

            # 手の位置を取得
            h, w, _ = frame.shape
            cx, cy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            px, py = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)

            cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
            cv2.circle(frame, (px, py), 20, (0, 255, 0), -1)

            # 人差し指と親指がくっついているか判定
            distance = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)

            if distance < 80:  # 閾値は適宜調整
                if not trajectories or not trajectories[-1]:
                    trajectories.append([(cx, cy)])  # 新しい軌跡の始点を追加
                else:
                    trajectories[-1].append((cx, cy))  # 現在の軌跡に座標を追加
            else:
                trajectories.append([])  # 人差し指と親指が離れている場合は新しい軌跡を始める

        for trajectory in trajectories:
            if len(trajectory) > 1:
                cv2.polylines(frame, [np.array(trajectory, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=5)

        # OpenCV画像をPySimpleGUIに反映
        disp_img = scale_to_height(frame, display_size[1])
        imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
        canvas_elem.update(data=imgbytes)

    window.close()

# 後片付け
cap.release()
cv2.destroyAllWindows()
