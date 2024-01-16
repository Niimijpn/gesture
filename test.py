import cv2
import mediapipe as mp
import PySimpleGUI as sg
import platform
import numpy as np
from PIL import ImageFont, ImageDraw, Image


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
    [sg.Button('SUB WINDOW', key="-SHOW_SUB_WINDOW-", size=(24, 1)),
    sg.Button('UNDO', key="-UNDO-", size=(10,1)),
    sg.Button('CLEAR', key="-CLEAR-", size=(10, 1))],
    [sg.Button('SAVE', key="-SAVE-", size=(10,1)),
    sg.Exit(key="-EXIT-", size=(10, 1))],
]

window = sg.Window('お絵描きアプリ', layout_main, resizable=True, finalize=True)
canvas_elem = window['-IMAGE-']
canvas = canvas_elem.Widget

layout_sub = [
    [sg.Button('サブ画面を閉じる', key="-CLOSE_SUB_WINDOW-", size=(20, 1))],
    [sg.Slider(range=(1, 30),
               key="-SLIDER-",
               default_value=5,
               size=(20, 15),
               orientation='horizontal',
               font=('Helvetica', 12))],
    [sg.Button('ペンの太さを更新', key="-UPDATE_SLIDER_VALUE-", size=(15, 1))],
    [sg.Text('色', size=(10, 1))],
    [sg.Button('赤', key="-RED-", size=(10, 1)),
     sg.Button('緑', key="-GREEN-", size=(10, 1)),
     sg.Button('青', key="-BLUE-", size=(10, 1)),
     sg.Button('黒', key="-BLACK-", size=(10, 1)),
     sg.Button('白', key="-WHITE-", size=(10, 1))],
]

window_sub = sg.Window('サブ画面', layout_sub, finalize=True)
window_sub.hide()  # 初めは非表示

# カメラ初期化
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(1)

# 手の検出モデルの初期化
with mp_hands.Hands(max_num_hands=2) as hands:
    trajectories = []  # 複数の軌跡を管理するリスト
    current_trajectory = []  # 現在の軌跡を管理するリスト
    slider_value = 5  # 初期値を設定
    col = [0, 0, 0]
    filrneme = "out.png"
    undo_stack = []

    while True:
        event, values = window.read(timeout=20)
        event_sub, values_sub,  = window_sub.read(timeout=20)

        if event in (sg.WINDOW_CLOSED, "-EXIT-"):
            break
        elif event == "-SHOW_SUB_WINDOW-":
            window_sub.un_hide()
        elif event_sub == "-CLOSE_SUB_WINDOW-":
            window_sub.hide()
        elif event == "-CLEAR-":
            trajectories = []  # サブ画面を閉じると軌跡もリセット
            current_trajectory = []  # クリアボタンを押すと現在の軌跡もリセット
        elif event_sub == "-UPDATE_SLIDER_VALUE-":
            slider_value = int(values_sub["-SLIDER-"])
        elif event_sub == "-RED-":
            col = [0, 0, 255]
        elif event_sub == "-GREEN-":
            col = [0, 255, 0]
        elif event_sub == "-BLUE-":
            col = [255, 0, 0]
        elif event_sub == "-BLACK-":
            col = [0, 0, 0]
        elif event_sub == "-WHITE-":
            col = [255, 255, 255]   
        elif event == "-UNDO-":
            if trajectories:
                undo_stack.append(trajectories.pop())  # 最後に描かれた軌跡を取り除き、UNDOスタックに追加
                window["-IMAGE-"].update(data=cv2.imencode('.png', scale_to_height(frame, display_size[1]))[1].tobytes())

        ret, frame = cap.read()
        
#        frame = cv2.flip(frame, 1)  # 画像反転

        if not ret:
            continue

        # フレームをBGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 手の検出
        results = hands.process(rgb_frame)

        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            
            # 手が複数検出されている場合、最初の手だけを取得
            hand_landmarks = results.multi_hand_landmarks[0]  
        
            h, w, _ = frame.shape
            px, py = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            
            if len(results.multi_hand_landmarks) > 1:
                
                # 手が複数検出されている場合、次の手を取得
                hand_landmarks_flag = results.multi_hand_landmarks[1]

                cx, cy = int(hand_landmarks_flag.landmark[4].x * w), int(hand_landmarks_flag.landmark[4].y * h)
                dx, dy = int(hand_landmarks_flag.landmark[8].x * w), int(hand_landmarks_flag.landmark[8].y * h)
                
                distance = np.sqrt((cx - dx) ** 2 + (cy - dy) ** 2)
                if distance < 60 :
                    # # テキストの描画
                    # text = "Hello, OpenCV!"
                    # org = (50, 150)
                    # fontFace = cv2.FONT_HERSHEY_SIMPLEX
                    # fontScale = 1.5
                    # color = (255, 255, 255)  # White color in BGR
                    # thickness = 2

                    # cv2.putText(frame, text, org, fontFace, fontScale, color, thickness)
                    
                    cv2.circle(frame, (px, py), 20, col, -1)
                    cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
                    if not current_trajectory:
                        current_trajectory.append((px, py))  # 新しい軌跡の始点を追加
                    else:
                        current_trajectory.append((px, py))  # 現在の軌跡に座標を追加
                else:
                    cv2.circle(frame, (px, py), 20, (0, 0, 0), -1)
                    cv2.circle(frame, (cx, cy), 20, (0, 0, 0), -1)
                    cv2.circle(frame, (dx, dy), 20, (0, 0, 0), -1)
                    if current_trajectory:
                        trajectories.append((current_trajectory.copy(), slider_value, col))  # 軌跡と太さを保存
                        current_trajectory = []  # 人差し指と親指が離れたら現在の軌跡をリセット

        for trajectory, thickness, color in trajectories:
            if len(trajectory) > 1:
                cv2.polylines(frame, [np.array(trajectory, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)

        if current_trajectory:
            cv2.polylines(frame, [np.array(current_trajectory, dtype=np.int32)], isClosed=False, color=col, thickness=slider_value)
                

        # OpenCV画像をPySimpleGUIに反映
        disp_img = scale_to_height(frame, display_size[1])        
        imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
        canvas_elem.update(data=imgbytes)
        
        
        if event == "-SAVE-":
            cv2.imwrite(filrneme, disp_img)

    window.close()

# 後片付け
cap.release()
cv2.destroyAllWindows()
