import cv2
import mediapipe as mp
import numpy as np


class HandWatchOverlay:
    def __init__(self, clock_image_path):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.clock_image = cv2.imread(clock_image_path, cv2.IMREAD_UNCHANGED)
        self.previous_distance = None

    def resize_watch(self, base_width, base_height, scale_factor):
        width = int(base_width * scale_factor)
        height = int(base_height * scale_factor)
        return cv2.resize(self.clock_image, (width, height))

    def overlay_watch(self, frame, clock_resized, wrist_x, wrist_y, width, height):
        clock_x = wrist_x - width // 2
        clock_y = wrist_y - height // 2

        if clock_x < 0:
            clock_x = 0
        if clock_y < 0:
            clock_y = 0
        if clock_x + width > frame.shape[1]:
            clock_x = frame.shape[1] - width
        if clock_y + height > frame.shape[0]:
            clock_y = frame.shape[0] - height

        # Убедимся, что размеры изображения часов соответствуют ожидаемым
        clock_resized = cv2.resize(clock_resized, (width, height))

        overlay = frame.copy()
        overlay[clock_y:clock_y + height, clock_x:clock_x + width] = clock_resized

        return overlay

    def calculate_distance(self, landmarks, idx1, idx2):
        x1, y1 = landmarks.landmark[idx1].x, landmarks.landmark[idx1].y
        x2, y2 = landmarks.landmark[idx2].x, landmarks.landmark[idx2].y
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_hands.process(frame_rgb)
                key = cv2.waitKey(1) & 0xFF

                desired_width = 50
                desired_height = 50

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                        wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                        distance = self.calculate_distance(hand_landmarks, 4, 8)
                        if self.previous_distance:
                            scale_factor = distance / self.previous_distance
                            desired_width = int(desired_width * scale_factor)
                            desired_height = int(desired_height * scale_factor)
                        else:
                            scale_factor = 1

                        clock_resized = self.resize_watch(desired_width, desired_height, scale_factor)
                        overlay = self.overlay_watch(frame, clock_resized, wrist_x, wrist_y, desired_width,
                                                     desired_height)
                        cv2.imshow("Frame", overlay)

                        self.previous_distance = distance

                else:
                    cv2.imshow("Frame", frame)

                if key == ord("q"):
                    break
            else:
                print("Ошибка: Не удалось захватить кадр из видеопотока.")
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Создание экземпляра класса и запуск приложения
watch_overlay = HandWatchOverlay('../watch3.jpeg')
watch_overlay.run()
