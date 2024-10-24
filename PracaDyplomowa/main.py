import mediapipe as mp
import numpy as np
import pyautogui as pag
import cv2

TEST = True
RIGHT_EYE = [386, 374]
LEFT_EYE = [159, 145]

RIGHT_EYE_FRAME_COUNT = 0
MIN_RIGHT_EYE_FRAME_COUNT = 30
LEFT_EYE_FRAME_COUNT = 0
MIN_LEFT_EYE_FRAME_COUNT = 30

maxValueLeftToNose, maxValueRightToNose, maxValueDownToNose, maxValueUpToNose, maxValueRightBlink, maxValueLeftBlink  = 0,0,0,0,0,0
screenWidth, screenHeight = pag.size()
mouseMovment = [0,0]
cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5,min_tracking_confidence=0.5)
draw = mp.solutions.drawing_utils



while 1:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    op = face.process(rgb)
    landmark_points = op.multi_face_landmarks
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for landmark in landmarks[4:5]:
            ## Punkt na nosie
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            if TEST:
                cv2.circle(frame, (x, y), 2, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

            ## Lewy i prawy punkt (policzki)
            x_left = int(landmarks[50].x * frame_w)
            y_left = int(landmarks[50].y * frame_h)
            x_right = int(landmarks[280].x * frame_w)
            y_right = int(landmarks[280].y * frame_h)

            ## Dolny punkt (broda)
            x_down = int(landmarks[0].x * frame_w)
            y_down = int(landmarks[0].y * frame_h)

            ## Górny punkt (między brwiami)
            x_up = int(landmarks[168].x * frame_w)
            y_up = int(landmarks[168].y * frame_h)

            ## Dystanse
            distance_up_nose = round(np.sqrt((x - x_up) ** 2 + (y - y_up) ** 2),2)
            distance_down_nose = round(np.sqrt((x - x_down) ** 2 + (y - y_down) ** 2),2)
            distance_left_nose = round(np.sqrt((x - x_left) ** 2 + (y - y_left) ** 2),2)
            distance_right_nose = round(np.sqrt((x - x_right) ** 2 + (y - y_right) ** 2),2)

            ## Wyświetlanie lini na ekranie
            if TEST:
                cv2.line(frame, (x_up, y_up), (x, y), (255, 0, 255), 2)
                cv2.line(frame, (x_down, y_down), (x, y), (125, 125, 255), 2)
                cv2.line(frame, (x_left, y_left), (x, y), (255, 255, 0), 2)
                cv2.line(frame, (x_right, y_right), (x, y), (0, 255, 255), 2)
                print("Lewy dystans",distance_left_nose," || Prawy dystans", distance_right_nose)
                print("Góra dystans", distance_up_nose, "|| Dół dystans", distance_down_nose)




            if maxValueUpToNose < distance_up_nose:
                maxValueUpToNose = distance_up_nose
            if maxValueDownToNose < distance_down_nose:
                maxValueDownToNose = distance_down_nose
            if maxValueLeftToNose < distance_left_nose:
                maxValueLeftToNose = distance_left_nose
            if maxValueRightToNose < distance_right_nose:
                maxValueRightToNose = distance_right_nose

            ## Ruch w góre z lekką akceleracja
            elif distance_up_nose < maxValueUpToNose * .6:
                if distance_up_nose < maxValueUpToNose * .45:
                    mouseMovment[1] = -6
                else:
                    mouseMovment[1] = -3
            ## Ruch w dół z lekką akceleracja
            elif distance_down_nose < maxValueDownToNose * .7:
                if distance_down_nose < maxValueDownToNose * .5:
                    mouseMovment[1] = 6
                else:
                    mouseMovment[1] = 3
            ## Ruch w lewo z lekką akceleracja
            if distance_left_nose < maxValueLeftToNose * .6:
                if distance_left_nose < maxValueLeftToNose * .3:
                    mouseMovment[0] = -6
                else:
                    mouseMovment[0] = -3
            ## Ruch w prawo z lekką akceleracja
            elif distance_right_nose < maxValueRightToNose * .6:
                if distance_right_nose < maxValueRightToNose * .3:
                    mouseMovment[0] = 6
                else:
                    mouseMovment[0] = 3

            for ex_data in RIGHT_EYE + LEFT_EYE:
                a = int(landmarks[ex_data].x * frame_w)
                b = int(landmarks[ex_data].y * frame_h)
                ab = [a,b]
                cv2.circle(frame, ab,color=(0, 255, 255), thickness=1,radius=1, lineType=cv2.LINE_AA)

            # wykrywanie mrugnięcia Prawy
            distance_right_eye_blink = round(np.sqrt(
                (landmarks[RIGHT_EYE[0]].x - landmarks[RIGHT_EYE[1]].x) ** 2 + (
                        landmarks[RIGHT_EYE[0]].y - landmarks[RIGHT_EYE[1]].y) ** 2), 2)
            if maxValueRightBlink < distance_right_eye_blink:
                maxValueRightBlink = distance_right_eye_blink

            if distance_right_eye_blink < maxValueRightBlink * .55:
                RIGHT_EYE_FRAME_COUNT = RIGHT_EYE_FRAME_COUNT + 1
                print(RIGHT_EYE_FRAME_COUNT)
            elif RIGHT_EYE_FRAME_COUNT > 0:
                RIGHT_EYE_FRAME_COUNT = 0

            if RIGHT_EYE_FRAME_COUNT == MIN_RIGHT_EYE_FRAME_COUNT:
                pag.click(button='right')
                RIGHT_EYE_FRAME_COUNT = 0

            # wykrywanie mrugnięcia Lewy
            distance_left_eye_blink = round(np.sqrt(
                (landmarks[LEFT_EYE[0]].x - landmarks[LEFT_EYE[1]].x) ** 2 + (
                            landmarks[LEFT_EYE[0]].y - landmarks[LEFT_EYE[1]].y) ** 2), 2)
            if maxValueLeftBlink < distance_left_eye_blink:
                maxValueLeftBlink = distance_left_eye_blink

            if distance_left_eye_blink < maxValueLeftBlink * .55:
                LEFT_EYE_FRAME_COUNT = LEFT_EYE_FRAME_COUNT + 1
                print(LEFT_EYE_FRAME_COUNT)
            elif LEFT_EYE_FRAME_COUNT > 0:
                LEFT_EYE_FRAME_COUNT = 0

            if LEFT_EYE_FRAME_COUNT == MIN_LEFT_EYE_FRAME_COUNT:
                pag.click(button='left')
                LEFT_EYE_FRAME_COUNT = 0
            #cv2.line(frame, (int(landmarks[159].x * frame_w),int(landmarks[159].y*frame_h)), (int(landmarks[153].x*frame_w), int(landmarks[153].y * frame_h)), (0, 0, 255), 2)


            ## Ruch myszki
            print(mouseMovment)
            pag.move(mouseMovment)
            mouseMovment = [0, 0]


    if op:
        cv2.imshow('kamera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break




'''
pomysły:

wargi jako scroll
opieranie głowy o barki jako scroll
mruganie jako prawy/lewy przycisk myszy
'''