import cv2
import numpy as np
import mediapipe as mp

# Список путей к изображениям часов
watch_images = [
    # "watch1.png",
    # "watch2.png",
    "watch3.jpeg",
    # ... Добавьте больше изображений часов
]

# Загрузка изображений часов
watches = []
for path in watch_images:
    watch = cv2.imread(path)
    if watch is None:
        print(f"Ошибка загрузки изображения: {path}")
    else:
        watches.append(watch)

# Инициализация захвата видео с веб-камеры
cap = cv2.VideoCapture(0)

# Инициализация рук медиапайпа
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Основной цикл
selected_watch = None

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

                # Наложение часов, если выбрано
                if selected_watch is not None:
                    watch_img = watches[selected_watch]

                    # Измените размер часов в зависимости от расстояния от запястья до среднего пальца (приблизительно)
                    middle_finger_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame.shape[1])

                    wrist_to_finger_dist = abs(wrist_x - middle_finger_x)
                    # Отрегулируйте коэффициент масштабирования по мере необходимости
                    watch_width = int(wrist_to_finger_dist * 1.5)
                    watch_height = int(watch_width * watch_img.shape[0] / watch_img.shape[1])
                    resized_watch = cv2.resize(watch_img, (watch_width, watch_height))

                    # Наложение часов с использованием альфа-наложения
                    # Отрегулируйте прозрачность по мере необходимости
                    alpha = 1.0
                    beta = 1 - alpha
                    overlay_region = frame[wrist_y - watch_height // 2:wrist_y + watch_height // 2,
                                     wrist_x - watch_width // 2:wrist_x + watch_width // 2]

                    frame[wrist_y - watch_height // 2:wrist_y + watch_height // 2,
                    wrist_x - watch_width // 2:wrist_x + watch_width // 2] = cv2.addWeighted(overlay_region, alpha,
                                                                                             resized_watch, beta, 0.0)

        # Отображение результата
        cv2.imshow("Frame", frame)

        # Проверить, нажата ли клавиша "s" для выбора часов
        if key == ord('s'):
            selected_watch = (selected_watch + 1) % len(watches)
            print(f"Выбраны часы {selected_watch + 1}")

        # Выход из цикла при нажатии клавиши "q"
        if key == ord("q"):
            break
    else:
        print("Error: Could not capture frame from video stream.")
        # Выйдите из цикла, если кадр не захвачен
        break

cap.release()
cv2.destroyAllWindows()
