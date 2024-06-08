import cv2
from gaze_tracking import GazeTracking
import mediapipe as mp
import time

gaze = GazeTracking()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
#webcam = cv2.VideoCapture("http://172.17.2.62:8001/video_feed")

blink_start_time = None
blinking_duration = 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        hand_detected = "Hand detected"
    else:
        hand_detected = ""

    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    gaze_text = ""

    if gaze.is_blinking():
        gaze_text = "Blinking"
        if blink_start_time is None:
            blink_start_time = time.time()
        else:
            blinking_duration = time.time() - blink_start_time
    else:
        blink_start_time = None
        blinking_duration = 0
        if gaze.is_right():
            gaze_text = "Looking right"
        elif gaze.is_left():
            gaze_text = "Looking left"
        elif gaze.is_center():
            gaze_text = "Looking center"

    if blinking_duration > 5:
        gaze_text = "Asleep"
        print("Asleep")

    cv2.putText(frame, gaze_text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display hand detection result
    cv2.putText(frame, hand_detected, (90, 250), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    # Display the frame
    cv2.imshow("Demo", frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) == 27:
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
