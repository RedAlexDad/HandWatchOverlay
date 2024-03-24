import cv2
import mediapipe as mp

# Инициализация захвата видео с веб-камеры
cap = cv2.VideoCapture(0)

# Инициализация рук медиапайпа
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    ret, frame = cap.read()

    # Проверьте, был ли кадр успешно прочитан
    if ret:
        # Преобразовать кадр в формат RGB для MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обнаруживать руки
        results = mp_hands.process(frame_rgb)

        # Получить код нажатой клавиши
        key = cv2.waitKey(1) & 0xFF

        # Если обнаружена рука
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Получить координаты ориентира на запястье
                wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                # Нарисуйте круг на запястье для визуализации
                cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 255, 0), -1)

        # Отображение результата
        cv2.imshow("Frame", frame)

        # Выход из цикла при нажатии клавиши "q"
        if key == ord("q"):
            break
    else:
        print("Ошибка: Не удалось захватить кадр из видеопотока.")
        # Выйдите из цикла, если кадр не захвачен
        break

cap.release()
cv2.destroyAllWindows()
