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

        # Желаемый размер часов (например, 100x100)
        desired_width = 50
        desired_height = 50

        # Изменение размера изображения с часами до желаемого размера
        clock_resized = cv2.resize(clock_image, (desired_width, desired_height))

        # Если обнаружена рука
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Получить координаты ориентира на запястье
                wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                # Установить положение часов на изображении
                clock_x = wrist_x - desired_width // 2  # координата x верхнего левого угла часов
                clock_y = wrist_y - desired_height // 2  # координата y верхнего левого угла часов

                # Проверка, чтобы часы не выходили за пределы кадра
                if clock_x < 0:
                    clock_x = 0
                if clock_y < 0:
                    clock_y = 0
                if clock_x + desired_width > frame.shape[1]:
                    clock_x = frame.shape[1] - desired_width
                if clock_y + desired_height > frame.shape[0]:
                    clock_y = frame.shape[0] - desired_height

                # Наложение часов на кадр
                overlay = frame.copy()
                overlay[clock_y:clock_y + desired_height, clock_x:clock_x + desired_width] = clock_resized

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