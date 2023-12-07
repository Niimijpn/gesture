import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=.5, min_tracking_confidence=.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGRからRGBに変換
        
        results = holistic.process(image) # 検出
   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGBからBGRに変換
        
#        # 顔
#        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
#                                 mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
#                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
#                                 )
        
        # 右手
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
                                 )
        
#        # 左手
#        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
#                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
#                                 )
#        
#        # 姿勢
#        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=1),
#                                 mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=1)
#                                 )
        
        cv2.imshow("Holistic Model Detections", image)

        # qを押したら終了する
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
            
cap.release()
cv2.destroyAllWindows()
