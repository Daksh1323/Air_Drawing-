import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            # Index finger tip
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            # Finger states
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y

            # Draw when only index finger is up
            if index_up and not middle_up:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Merge drawing and webcam
    frame = cv2.add(frame, canvas)

    cv2.putText(frame, "Press 'c' to clear | 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("Finger Drawing", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
