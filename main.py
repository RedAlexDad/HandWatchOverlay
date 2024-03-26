import cv2
import mediapipe as mp
import numpy as np
import os

class HandWatchOverlay:
    def __init__(self, watchs_folder_path: any, desired_width: int = 50, desired_height: int = 50) -> None:
        '''
        Инициализация класса HandWatchOverlay.
        :param watchs_folder_path: Путь к папке с изображениями часов.
        :type watchs_folder_path: str
        :param: desired_width: Желаемая ширина часов на кадре.
        :type: desired_height: int
        :param: desired_height: Желаемая высота часов на кадре.
        :type: desired_height: int
        :return: None
        :rtype: None
        '''
        self.cap = cv2.VideoCapture(0)
        # min_detection_confidence: Минимальная уверенность (confidence), требуемая для обнаружения руки.
        # min_tracking_confidence: Минимальная уверенность (confidence), требуемая для отслеживания руки.
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Чтение изображения с альфа-каналом
        self.clock_images = [cv2.imread(os.path.join(watchs_folder_path, img_name), cv2.IMREAD_UNCHANGED) for img_name in os.listdir(watchs_folder_path)]
        self.current_clock_index = 0
        self.previous_distance = None
        self.scale_factor_history = []
        self.desired_width = desired_width
        self.desired_height = desired_height

    def switch_clock_image(self) -> None:
        '''
        Переключает текущее изображение часов на следующее в списке изображений.
        :return: None
        :rtype: None
        '''
        self.current_clock_index = (self.current_clock_index + 1) % len(self.clock_images)
        clock_name = f"Clock image {self.current_clock_index + 1}"
        cv2.putText(self.frame, clock_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("Switched to clock image", clock_name)

    def calculate_average_scale_factor(self) -> float:
        '''
        Вычисляет средний масштабный коэффициент из истории. Если история пуста, возвращает 1.0.
        :return: float Средний масштабный коэффициент.
        :rtype: float
        '''
        if len(self.scale_factor_history) > 0:
            return sum(self.scale_factor_history) / len(self.scale_factor_history)
        else:
            return 1.0

    def resize_watch(self, clock_image: np.ndarray, base_width: int, base_height: int, scale_factor: float=1.0) -> np.ndarray:
        '''
        Изменяет размер изображения часов. Если scale_factor < 0, возвращает исходное изображение.
        :param: clock_image: Изображение часов.
        :type: np.ndarray
        :param: base_width: Исходная ширина часового изображения.
        :type: int
        :param: base_height: Исходная высота часового изображения.
        :type: int
        :param: scale_factor: Масштабный коэффициент для изменения размера часового изображения (по умолчанию 1.0).
        :type: float
        :return: Изображение часов после изменения размера.
        :rtype: np.ndarray
        '''
        if isinstance(clock_image, np.ndarray):
            if scale_factor < 0:
                width = int(base_width * scale_factor)
                height = int(base_height * scale_factor)
                return cv2.resize(clock_image, (width, height), interpolation=cv2.INTER_AREA)
            else:
                return cv2.resize(clock_image, (base_width, base_height), interpolation=cv2.INTER_AREA)
        else:
            print("Ошибка: Изображение часов не является массивом numpy.")
            return None

    def overlay_watch(self, frame: np.ndarray, clock_resized, wrist_x: int, wrist_y: int, width: int, height: int) -> np.ndarray:
        '''
        :param: frame: Входной кадр из видеопотока.
        :type: frame: np.ndarray
        :param: clock_resized: Изображение часов после изменения размера.
        :param: wrist_x: Координата x запястья руки на кадре.
        :type: wrist_x: int
        :param: wrist_y: Координата y запястья руки на кадре.
        :type: wrist_y: int
        :param: width: Ширина области, на которую будет наложено изображение часов.
        :type: width: int
        :param: height: Высота области, на которую будет наложено изображение часов.
        :type: height: int
        :return: Наложенное изображение часов.
        :rtype: np.ndarray
        '''
        clock_x = wrist_x - width // 2
        clock_y = wrist_y - height // 2
        clock_x, clock_y = self.adjust_coordinates(clock_x, clock_y, width, height, frame.shape[1], frame.shape[0])

        if clock_resized.shape[2] == 4:  # Проверка на наличие альфа-канала
            alpha = clock_resized[:, :, 3] / 255.0

            # Создание маски для наложения
            overlay = frame.copy()
            mask = np.ones_like(frame, dtype=float)
            mask[clock_y:clock_y + height, clock_x:clock_x + width] = 1.0 - alpha[:, :, np.newaxis]

            # Наложение изображения с учётом альфа-канала
            overlay[clock_y:clock_y + height, clock_x:clock_x + width] = (
                    alpha[:, :, np.newaxis] * clock_resized[:, :, :3] +
                    mask[clock_y:clock_y + height, clock_x:clock_x + width] * overlay[clock_y:clock_y + height,
                                                                              clock_x:clock_x + width])
        else:
            # Простое наложение изображения без учёта прозрачности
            overlay = frame.copy()
            overlay[clock_y:clock_y + height, clock_x:clock_x + width] = clock_resized

        return overlay

    def adjust_coordinates(self, clock_x: int, clock_y: int, width: int, height: int, frame_width: int, frame_height: int) -> tuple:
        '''
        Функция корректирует координаты изображения часов, если они выходят за пределы кадра.
        :param: clock_x: Координата x изображения часов.
        :type: clock_x: int
        :param: clock_y: Координата y изображения часов.
        :type: clock_y: int
        :param: width: Ширина изображения часов.
        :type: width: int
        :param: height: Высота изображения часов.
        :type: height: int
        :param: frame_width: Ширина кадра.
        :type: frame_width: int
        :param: frame_height: Высота кадра.
        :type: frame_height: int
        :return: Корректированные координаты изображения часов.
        :rtype: tuple
        '''

        if clock_x < 0:
            clock_x = 0
        if clock_y < 0:
            clock_y = 0
        if clock_x + width > frame_width:
            clock_x = frame_width - width
        if clock_y + height > frame_height:
            clock_y = frame_height - height

        return clock_x, clock_y

    def calculate_distance(self, landmarks: any, idx1: int, idx2: int) -> float:
        '''
        Вычисление расстояния между двумя ключевыми точками.
        :param landmarks: Ключевые точки руки, обнаруженные моделью MediaPipe Hands.
        :type landmarks: mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        :param idx1: Индекс первой ключевой точки для вычисления расстояния.
        :type idx1: int
        :param idx2: Индекс второй ключевой точки для вычисления расстояния.
        :type idx2: int
        :return: Расстояние между ключевыми точками.
        :rtype: float
        '''
        x1, y1 = landmarks.landmark[idx1].x, landmarks.landmark[idx1].y
        x2, y2 = landmarks.landmark[idx2].x, landmarks.landmark[idx2].y
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def process_frame(self, frame: np.ndarray) -> int:
        '''
        Обрабатывает каждый кадр видеопотока.
        :param frame: Входной кадр из видеопотока.
        :type frame: np.ndarray
        :return: Код клавиши, которая была нажата (если была).
        :rtype: int
        '''
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(frame_rgb)
        key = cv2.waitKey(1) & 0xFF

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                distance = self.calculate_distance(hand_landmarks, 4, 8)
                if self.previous_distance:
                    scale_factor = distance * 7
                    self.scale_factor_history.append(scale_factor)
                    if len(self.scale_factor_history) > 10:
                        self.scale_factor_history.pop(0)
                    average_scale_factor = self.calculate_average_scale_factor()
                    desired_width = int(self.desired_width * average_scale_factor)
                    desired_height = int(self.desired_height * average_scale_factor)

                    clock_resized = self.resize_watch(self.clock_images[self.current_clock_index], desired_width,
                                                      desired_height, average_scale_factor)

                    overlay = self.overlay_watch(frame, clock_resized, wrist_x, wrist_y, desired_width,
                                                 desired_height)
                    cv2.imshow("Frame", overlay)

                self.previous_distance = distance
        else:
            cv2.imshow("Frame", frame)

        # Обработка события нажатия клавиши "S" для смены изображения часов
        if key == ord("s"):
            self.current_clock_index = (self.current_clock_index + 1) % len(self.clock_images)
            print(f"Изображение часов изменено на {self.current_clock_index + 1}")

        return key

    def process_video(self, video_path: np.ndarray) -> None:
        '''
        Обрабатывает видеофайл.
        :param: video_path: Путь к видеофайлу.
        :type: video_path: str
        :return: None
        :rtype: None
        '''
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Ошибка: Не удалось открыть видеофайл.")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            key = self.process_frame(frame)

            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        '''
        Запуск приложения в реальном времени.
        :return: None
        :rtype: None
        '''
        while True:
            ret, frame = self.cap.read()

            if ret:
                key = self.process_frame(frame)
                if key == ord("q"):
                    break
            else:
                print("Ошибка: Не удалось захватить кадр из видеопотока.")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Создание экземпляра класса
    watch_overlay = HandWatchOverlay(watchs_folder_path='watches/')
    # Запуск приложения в реальном времени с веб-камерой
    watch_overlay.run()

    # Запуск приложения в реальном времени без веб-камеры (выполняется в отдельном процессе, видео загружается)
    try:
        watch_overlay_1 = HandWatchOverlay('watches/', desired_width=100, desired_height=100)
        # Замените 'your_video.mp4' на путь к вашему видеофайлу
        watch_overlay_1.process_video('example1.mov')
    except Exception as error:
        print(error) # Вывод ошибки в консоль
