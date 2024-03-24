import cv2
import mediapipe as mp


class HandWatchOverlay:
    def __init__(self, clock_image_path):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.clock_image = cv2.imread(clock_image_path, cv2.IMREAD_UNCHANGED)

    def resize_watch(self, desired_width, desired_height):
        '''
        :param:
        desired_width: Желаемая ширина изображения часов.
        desired_height: Желаемая высота изображения часов.
        '''
        return cv2.resize(self.clock_image, (desired_width, desired_height))

    def overlay_watch(self, frame, clock_resized, wrist_x, wrist_y, desired_width, desired_height):
        '''
        :param:
        frame: Кадр, на котором будет наложено изображение часов.
        clock_resized: Измененное размером изображение часов.
        wrist_x: Координата X запястья.
        wrist_y: Координата Y запястья.
        desired_width: Желаемая ширина изображения часов.
        desired_height: Желаемая высота изображения часов.
        '''
        clock_x = wrist_x - desired_width // 2
        clock_y = wrist_y - desired_height // 2

        if clock_x < 0:
            clock_x = 0
        if clock_y < 0:
            clock_y = 0
        if clock_x + desired_width > frame.shape[1]:
            clock_x = frame.shape[1] - desired_width
        if clock_y + desired_height > frame.shape[0]:
            clock_y = frame.shape[0] - desired_height

        overlay = frame.copy()
        overlay[clock_y:clock_y + desired_height, clock_x:clock_x + desired_width] = clock_resized

        return overlay

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_hands.process(frame_rgb)
                key = cv2.waitKey(1) & 0xFF

                desired_width = 50
                desired_height = 50

                clock_resized = self.resize_watch(desired_width, desired_height)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                        wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                        overlay = self.overlay_watch(frame, clock_resized, wrist_x, wrist_y, desired_width,
                                                     desired_height)
                        cv2.imshow("Frame", overlay)

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
