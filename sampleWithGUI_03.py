# ステップ1. インポート
import PySimpleGUI as sg  # PySimpleGUIをsgという名前でインポート
import os  # OS依存の操作（Pathやフォルダ操作など）用ライブラリのインポート
import numpy as np  # numpyのインポート
import cv2  # OpenCV（python版）のインポート
import mediapipe as mp  #mediapipeのインクルード

# ---------  関数群  ----------

# モザイク処理関数
def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    # src.shape[:2][::-1]は要素を先頭から2つ取ってきて逆順に並び替える操作
    # nparrayは行・列の並びなので，例えば画像サイズが（640,480）の時
    # shapeは shape[0]=480, shape[1]=640（480, 640）となる．
    # resizeに渡す第２引数は（横640，縦480）なので，[::-1]で逆順に並べ替える
    # （参考）https://qiita.com/tanuk1647/items/276d2be36f5abb8ea52e

def mozaic_area(src, box, ratio=0.1):
    dst = src.copy()
    # 画像外へのアクセスチェック
    if box[0] < 0:
        box[0] = 0;
    if box[1] < 0:
        box[1] = 0;
    if box[2] >= src.shape[1]:
        box[2] = src.shape[1];
    if box[3] >= src.shape[0]:
        box[3] = src.shape[0];
    dst[box[1]:box[3], box[0]:box[2]] = mosaic(dst[box[1]:box[3], box[0]:box[2]], ratio)
    
    return dst

# アイコンを読み込む関数
def load_image(path):
    icon = cv2.imread(path, -1)
    if icon.data == 0:
        print('画像が読み込めませんでした')
    return icon

# 画像をリサイズする関数
def img_resize(img, scale):
    h, w  = img.shape[:2]
    img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

# 画像を保存する関数
def save_image(img):
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = "./" + date + ".png"
    cv2.imwrite(path, img) # ファイル保存

# 画像を合成する関数
def merge_images(bg, fg_alpha, s_x, s_y):
    alpha = fg_alpha[:,:,3]  # アルファチャンネルだけ抜き出す(要は2値のマスク画像)
    alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # grayをBGRに
    alpha = alpha / 255.0    # 0.0〜1.0の値に変換
    
    fg = fg_alpha[:,:,:3]
    
    f_h, f_w, _ = fg.shape # アルファ画像の高さと幅を取得
    b_h, b_w, _ = bg.shape # 背景画像の高さを幅を取得
    
    # 画像の大きさと開始座標を表示
#    print("f_w:{} f_h:{} b_w:{} b_h:{} s({}, {})".format(f_w, f_h, b_w, b_h, s_x, s_y))
    
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] * (1.0 - alpha)).astype('uint8') # アルファ以外の部分を黒で合成
    bg[s_y:f_h+s_y, s_x:f_w+s_x] = (bg[s_y:f_h+s_y, s_x:f_w+s_x] + (fg * alpha)).astype('uint8')  # 合成
    
    return bg

# 画像リサイズ関数（高さが指定した値になるようにリサイズ (アスペクト比を固定)）
def scale_to_height(img, height):
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))
	
    return dst

# ---- 顔認識エンジンセット ----
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_detection = mp.solutions.face_detection

# ---- 大域変数 ----
display_size = (400, 300)  # ディスプレイサイズ
isOpened = 0   # カメラがオープンになっているかどうかのフラグ
isWriting = 0   # 録画ON/OFF
movie_name = 'output.mp4'
IMAGE_PATH = "./nc73730.png"  # 画像パス

# ステップ2. デザインテーマの設定
sg.theme('DarkTeal7')

# ステップ3. ウィンドウの部品とレイアウト
layout = [
		  [sg.Text('カメラ')],
		  [sg.Button('カメラ', key='camera')],
          [sg.Image(filename='', size=display_size, key='-input_image-')],
          [sg.Text('ランドマークの表示', size=(15, 1)), sg.Combo(('あり', 'なし'), default_value='あり', size=(5, 1), key='landmark'),
          sg.Text('モザイクの表示', size=(15, 1)), sg.Combo(('あり', 'なし'), default_value='なし', size=(5, 1), key='mozaic'),
          sg.Text('VTuberの表示', size=(15, 1)), sg.Combo(('あり', 'なし'), default_value='なし', size=(5, 1), key='vtuber')],
          [sg.Button('動画保存', key='save'), sg.Button('終了', key='exit')],
		  [sg.Output(size=(80,10))]
		  ]

# ステップ4. ウィンドウの生成
window = sg.Window('人体を認識するツール', layout, location=(400, 20))

# ステップ5. カメラ，mediapipeの初期設定
#cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:
    # ステップ6. イベントループ
    while True:
        event, values = window.read(timeout=10)
        
        if event in (None, 'exit'): #ウィンドウのXボタンまたは”終了”ボタンを押したときの処理
            break
        
        if event == 'camera':  #「カメラ」ボタンが押された時の処理
            print("Camera Open")
            cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する
            isOpened, orig_img = cap.read()
            if isOpened:  # 正常にフレームを読み込めたら
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("Frame size = " + str(orig_img.shape))
                frame_size = (width, height)
                # Vtuber画像の読み込み
                icon = load_image(IMAGE_PATH)
                # 表示用に画像を固定サイズに変更（大きい画像を入力した時に認識ボタンなどが埋もれないように）
                disp_img = scale_to_height(orig_img, display_size[1])
                # 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
                imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
                # ウィンドウへ表示
                window['-input_image-'].update(data=imgbytes)
            else:
                print("Cannot capture a frame image")

        if isOpened == 1:
            # ---- フレーム読み込み ----
            ret, frame = cap.read()
            if ret:  # 正常にフレームを読み込めたら
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGRからRGBに変換
                results = holistic.process(image) # 検出
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGBからBGRに変換
            
                if(values['mozaic'] == 'あり' or values['vtuber'] == 'あり'):
                    # 顔検出セット
                    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                        faces = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  #顔検出実行
                        if faces.detections:
#                            print(faces.detections)
                            for detection in faces.detections:
                                bb = detection.location_data.relative_bounding_box
                                # bounding_box: xmin, ymin, width, height （注）0~1に正規化された値．実際の座標は画像のサイズを掛けること
                                left = int(bb.xmin * width)
                                top = int(bb.ymin * height)
                                right = left + int(bb.width * width)
                                bottom = top + int(bb.height * height)
                                box = (left, top, right, bottom)
#                                print(box)
                                if(values['mozaic'] == 'あり'):
                                    image = mozaic_area(image, box, 0.05)
                                if(values['vtuber'] == 'あり'):
#                                    print('vtuber')
                                    icon = img_resize(icon, bb.height*height/icon.shape[0])
                                    # icon画像の中心をランドマーク（鼻）に対応づける
                                    nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
                                    x = int(nose.x*width) - int(icon.shape[1]/2)
                                    y = int(nose.y*height) - int(icon.shape[0]/2)
                                    if (0 <= y) and (y <= (height-int(icon.shape[0]))) and (0 <= x) and (x <= (width-int(icon.shape[1]))):
                                        # 画面の範囲内だったら画像を合成
                                        image = merge_images(image, icon, x, y)
                                    
#                                mp_drawing.draw_detection(image, detection)  # 検出したランドマークの表示（要らなければコメントに）
                                
                if(values['landmark'] == 'あり'):
                    # 顔
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
                                             )
                    
                    # 右手
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
                                             )
                    
                    # 左手
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
                                             )

                    # 姿勢
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
                                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
                                             )
                # 表示用に画像を固定サイズに変更
                disp_img = scale_to_height(image, display_size[1])
                # 表示用画像データをPNG（１画素４バイト）に変換し，１次元配列（バイナリ形式）に入れ直し
                imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
                # ウィンドウへ表示
                window['-input_image-'].update(data=imgbytes)


        if event == 'save': #「保存」ボタンが押されたときの処理
            # VideoWriterでムービー書き出し処理を書く
            # 「保存」ボタンをもう一度押したら停止するようにする
            if isWriting == 0:
                print("Write movie -> " + movie_name)
                isWriting = 1
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(movie_name, codec, 10.0, frame_size)
            elif isWriting == 1:
                print("Writing End")
                writer.release()
                isWriting = 0
        
        if isWriting == 1:  # if isOpened == 1:の中に入れても良い
            writer.write(image)

cap.release()
window.close()
