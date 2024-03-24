import cv2
import dlib
import numpy as np


class HandWatchOverlay:
    def __init__(self, clock_image_path):
        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.clock_image = cv2.imread(clock_image_path, cv2.IMREAD_UNCHANGED)

    def overlay_watch(self, frame, wrist_x, wrist_y, width, height):
        clock_x = wrist_x - width // 2
        clock_y = wrist_y - height // 2

        overlay = frame.copy()

        # Проверка на наличие альфа-канала в изображении часов
        if self.clock_image.shape[2] == 4:  # RGBA изображение
            # Разделение изображения на каналы BGR и альфа
            clock_bgr = self.clock_image[:, :, :3]
            alpha_mask = self.clock_image[:, :, 3]

            # Наложение изображения с альфа-каналом
            overlay_area = overlay[clock_y:clock_y + height, clock_x:clock_x + width]
            overlay_area = overlay_area.astype('float64')  # Приведение к типу данных float64
            overlay_area *= (255 - alpha_mask) / 255  # Уменьшаем значения пикселей в соответствии с альфа-каналом
            overlay_area += (alpha_mask * clock_bgr) / 255  # Добавляем значения изображения часов
            overlay_area = np.clip(overlay_area, 0, 255).astype(
                np.uint8)  # Приводим обратно к uint8 и обрезаем значения
            overlay[clock_y:clock_y + height, clock_x:clock_x + width] = overlay_area
        else:
            # Если альфа-канала нет, накладываем изображение как обычно
            overlay[clock_y:clock_y + height, clock_x:clock_x + width] = self.clock_image

        return overlay

    def run(self):
        while True:
            ret, frame = self.cap.read()

            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                for face in faces:
                    landmarks = self.predictor(gray, face)
                    wrist_x = landmarks.part(16).x
                    wrist_y = landmarks.part(16).y

                    # Ширина и высота для часов
                    width = self.clock_image.shape[1]
                    height = self.clock_image.shape[0]

                    overlay = self.overlay_watch(frame, wrist_x, wrist_y, width, height)
                    cv2.imshow("Frame", overlay)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                print("Ошибка: Не удалось захватить кадр из видеопотока.")
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Создание экземпляра класса и запуск приложения
watch_overlay = HandWatchOverlay('../watch1.png')
watch_overlay.run()
