import cv2
import mediapipe as mp

# Инициализация захвата видео с веб-камеры
cap = cv2.VideoCapture(0)

# Инициализация рук медиапайпа
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Загрузка изображения с часами
# clock_image = cv2.imread('watch1.png', cv2.IMREAD_UNCHANGED)
clock_image = cv2.imread('watch3.jpeg', cv2.IMREAD_UNCHANGED)

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

                # Изменение размера изображения с часами до размеров кадра
                clock_resized = cv2.resize(clock_image, (frame.shape[1], frame.shape[0]))

                # Наложение изображения с часами на кадр
                overlay = frame.copy()
                overlay[0:frame.shape[0], 0:frame.shape[1]] = clock_resized

                # Отображение измененного кадра в том же окне
                cv2.imshow("Frame", overlay)

        else:
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