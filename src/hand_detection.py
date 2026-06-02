import cv2 # type: ignore
import mediapipe as mp # type: ignore
import os
import time


SAVE_DIR = "assets/screenshots"


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def get_finger_states(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def get_gesture(fingers):
    if fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    else:
        return "Unknown"


def run_hand_gesture(camera_index=0):
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    window_name = "Hand Gesture"

    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        finger_count = 0
        gesture = "No Hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                fingers = get_finger_states(hand_landmarks)
                finger_count = sum(fingers)
                gesture = get_gesture(fingers)

                if gesture == "Open Palm":
                    current_time = time.time()
                    if current_time - last_capture_time > 2:
                        filename = os.path.join(
                            SAVE_DIR,
                            f"img_{int(current_time)}.jpg"
                        )
                        cv2.imwrite(filename, frame)
                        print(f"Image saved: {filename}")
                        last_capture_time = current_time

        cv2.putText(frame,
                    f"Gesture: {gesture}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Fingers: {finger_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.imshow(window_name, frame)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hand_gesture()