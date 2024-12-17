import mediapipe as mp
import numpy as np
import pyautogui as pag
import cv2
import json
import sys

# WARTOSCI KONFIGURACYJNE
TEST = True # wyświtlaj wartości oraz punkty/linie na ekranie
'''
MIN_RIGHT_EYE_FRAME_COUNT = 30 # Czas zaliczenia reakcji do P/L przycisku myszy
MIN_LEFT_EYE_FRAME_COUNT = 30
MIN_DABBLE_CLICK_EYE_FRAME_COUNT = 10
Config_Moving_Up = 0.6 # Wartość konfiguracjynja przesuwająca kursor w góre [zakres od 0 do 1]
Config_Moving_Down = 0.6 # Wartość konfiguracjynja przesuwająca kursor w dół [zakres od 0 do 1]
Config_Moving_Left = 0.6 # Wartość konfiguracjynja przesuwająca kursor w lewo [zakres od 0 do 1]
Config_Moving_Right = 0.6 # Wartość konfiguracjynja przesuwająca kursor w prawo [zakres od 0 do 1]
Moving_value = [-3, -6, 3, 6, -3, -6, 3, 6] # Wartości szybkościu poruszania się kursora myszy (odpowiednio po dwie
#                                                 # wartości (góra, dół, lewo, prawo). pierwsza wartość pary to podstawowe
#                                                     # szybkość a druga przyspieszenie
'''
try:
    # Próbujemy otworzyć i wczytać plik
    with open('config.json', 'r') as file:
        config = json.load(file)

        MIN_RIGHT_EYE_FRAME_COUNT = config["eye_frame_count"]["min_right_eye"]
        MIN_LEFT_EYE_FRAME_COUNT = config["eye_frame_count"]["min_left_eye"]
        MIN_DABBLE_CLICK_EYE_FRAME_COUNT = config["eye_frame_count"]["min_dabble_click_eye"]

        Config_Moving_Up = config["cursor_movement"]["config_moving_up"]
        Config_Moving_Down = config["cursor_movement"]["config_moving_down"]
        Config_Moving_Left = config["cursor_movement"]["config_moving_left"]
        Config_Moving_Right = config["cursor_movement"]["config_moving_right"]

        Moving_value = config["moving_value"]

except Exception as e:
    print("błąd podczas odczytu pliku konfiguracyjnego")
    sys.exit(1)


## WARTOSCI STAŁE
# Inicjacja wartości
RIGHT_EYE_FRAME_COUNT = 0
LEFT_EYE_FRAME_COUNT = 0
DABBLE_CLICK_EYE_FRAME_COUNT = [0,0]

RIGHT_EYE = [386, 374] # pozycje górna i dolna powiek P/L oka
LEFT_EYE = [159, 145]
maxValueLeftToNose, maxValueRightToNose, maxValueDownToNose, maxValueUpToNose, maxValueRightBlink, maxValueLeftBlink  = 0,0,0,0,0,0
screenWidth, screenHeight = pag.size()
mouseMovement = np.array([0,0]) # Inicjacja wartości przesuwania kurosora

def set_mouse_movement(current_distance, max_distance_value, config_value, axis, direction_of_movement) -> list[int]:
    """
    :param current_distance: (float)
    :param max_distance_value: (float)
    :param config_value: (float) in between 0 and 1
    :param axis: A value in range [0, 1]
    :param direction_of_movement: A value in range [0, 1, 2, 3]  (up, down, left, right)
    :return: list[int]: A one-dimensional list containing two integer values: [int, int]
    """
    movement = [0,0]
    if current_distance < max_distance_value * config_value:
        if current_distance < max_distance_value * (config_value / 2):
            movement[axis] = Moving_value[direction_of_movement * 2 + 1]
        else:
            movement[axis] = Moving_value[direction_of_movement * 2]
    return np.array(movement)


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    pag.FAILSAFE = False  # wyłączenie failsafe
    facemesh = mp.solutions.face_mesh
    face = facemesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
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
            landmark = landmarks[4]
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
            x_down = int(landmarks[16].x * frame_w)
            y_down = int(landmarks[16].y * frame_h)

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

            maxValueUpToNose = max(maxValueUpToNose, distance_up_nose)
            maxValueDownToNose = max(maxValueDownToNose, distance_down_nose)
            maxValueLeftToNose = max(maxValueLeftToNose, distance_left_nose)
            maxValueRightToNose = max(maxValueRightToNose, distance_right_nose)

            ## Ruch w góre z lekką akceleracja
            mouseMovement = mouseMovement + set_mouse_movement(distance_up_nose, maxValueUpToNose, Config_Moving_Up, 1, 0)
            ## Ruch w dół z lekką akceleracja
            mouseMovement = mouseMovement + set_mouse_movement(distance_down_nose, maxValueDownToNose, Config_Moving_Down, 1, 1)
            ## Ruch w lewo z lekką akceleracja
            mouseMovement = mouseMovement + set_mouse_movement(distance_left_nose, maxValueLeftToNose, Config_Moving_Left,0,2)
            ## Ruch w prawo z lekką akceleracja
            mouseMovement = mouseMovement + set_mouse_movement(distance_right_nose, maxValueRightToNose, Config_Moving_Right,0,3)

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
                DABBLE_CLICK_EYE_FRAME_COUNT[1] = DABBLE_CLICK_EYE_FRAME_COUNT[1] + 1
                print(RIGHT_EYE_FRAME_COUNT)
            elif RIGHT_EYE_FRAME_COUNT > 0:
                RIGHT_EYE_FRAME_COUNT = 0
                DABBLE_CLICK_EYE_FRAME_COUNT[1] = 0

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
                DABBLE_CLICK_EYE_FRAME_COUNT[0] = DABBLE_CLICK_EYE_FRAME_COUNT[0] + 1
                print(LEFT_EYE_FRAME_COUNT)
            elif LEFT_EYE_FRAME_COUNT > 0:
                LEFT_EYE_FRAME_COUNT = 0
                DABBLE_CLICK_EYE_FRAME_COUNT[0] = 0

            if LEFT_EYE_FRAME_COUNT == MIN_LEFT_EYE_FRAME_COUNT:
                pag.click(button='left')
                LEFT_EYE_FRAME_COUNT = 0

            if DABBLE_CLICK_EYE_FRAME_COUNT[0] == MIN_DABBLE_CLICK_EYE_FRAME_COUNT and DABBLE_CLICK_EYE_FRAME_COUNT[1] == MIN_DABBLE_CLICK_EYE_FRAME_COUNT:
                pag.click(button='left',clicks=2)
                DABBLE_CLICK_EYE_FRAME_COUNT = [0,0]

            if TEST:
                print("Lewe oko dystans",distance_left_eye_blink,"\nPrawy dystans", distance_right_eye_blink,"\n")
                ## Ruch myszki
                print(mouseMovement)
                print("DABBLE_CLICK_EYE_FRAME_COUNT:", DABBLE_CLICK_EYE_FRAME_COUNT)
            pag.move(*mouseMovement)
            mouseMovement = np.array([0,0])


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