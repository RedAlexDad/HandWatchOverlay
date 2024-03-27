# HandWatchOverlay

Репозиторий **HandWatchOverlay** представляет собой программное обеспечение, разработанное для наложения изображения
часов на видеопоток с помощью алгоритмов компьютерного зрения. Это приложение предназначено для работы с камерой или
видеофайлом, обнаруживает руку на изображении с помощью библиотеки MediaPipe, а затем налагает изображение часов на
запястье. Программа поддерживает изображения часов с прозрачным фоном в формате PNG, что позволяет сохранять
естественный вид руки и окружающего фона.

## Основные функции

- Использование технологии OpenCV.
- Обнаружение руки и ее ключевых точек с помощью библиотеки MediaPipe.
- Наложение изображения часов на обнаруженное запястье с сохранением пропорций и прозрачности.
- Возможность работы как с видеопотоком, так и с видеофайлом.

### Горячие клавиатуры:

- Переключить на другие виды часы `S`
- Прервать работу - `Q`

## Демонстрация

<img src="images/img.png" alt="img.png" width="50%" height="50%">
<img src="images/img_1.png" alt="img_1.png" width="50%" height="50%">

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/RedAlexDad/HandWatchOverlay.git
```

## Диаграмма
```mermaid
sequenceDiagram
    participant User
    participant HandWatchOverlay
    participant cv2.VideoCapture
    participant mediapipe.hands.Hands
    participant cv2.imread
    participant cv2.resize
    participant cv2.putText
    participant cv2.imshow
    participant cv2.waitKey
    participant cv2.VideoCapture.process_frame
    participant cv2.VideoCapture.release
    participant cv2.destroyAllWindows

    User ->> HandWatchOverlay: Создание экземпляра класса HandWatchOverlay с папкой изображений часов
    HandWatchOverlay ->> cv2.VideoCapture: Захват кадра с веб-камеры
    cv2.VideoCapture ->> mediapipe.hands.Hands: Обработка видеопотока (модель MediaPipe Hands)
    mediapipe.hands.Hands -->> cv2.VideoCapture: Результаты обработки видеопотока
    cv2.VideoCapture ->> HandWatchOverlay: Обработка кадра
    HandWatchOverlay ->> cv2.imread: Чтение изображений часов
    cv2.imread -->> HandWatchOverlay: Изображения часов
    HandWatchOverlay ->> cv2.resize: Изменение размера изображений часов
    cv2.resize -->> HandWatchOverlay: Измененные изображения часов
    HandWatchOverlay ->> cv2.putText: Добавление названия изображения часов на кадр
    cv2.putText -->> HandWatchOverlay: Кадр с названием изображения часов
    HandWatchOverlay ->> cv2.imshow: Отображение обработанного кадра
    cv2.imshow -->> HandWatchOverlay: Отображенный кадр
    HandWatchOverlay ->> cv2.waitKey: Ожидание нажатия клавиши
    cv2.waitKey -->> HandWatchOverlay: Код нажатой клавиши
    HandWatchOverlay ->> cv2.VideoCapture.process_frame: Обработка следующего кадра
    cv2.VideoCapture.process_frame -->> cv2.VideoCapture: Обработанный кадр
    cv2.VideoCapture.process_frame -->> User: Нажатая клавиша (если была)
    cv2.waitKey -->> User: Код нажатой клавиши
    User ->> HandWatchOverlay: Нажатие клавиши "s" для смены изображения часов
    HandWatchOverlay ->> HandWatchOverlay: Смена изображения часов
    HandWatchOverlay ->> cv2.VideoCapture: Захват следующего кадра с веб-камеры
    cv2.VideoCapture -->> cv2.VideoCapture: Следующий кадр с веб-камеры
    cv2.VideoCapture ->> HandWatchOverlay: Обработка кадра
    HandWatchOverlay ->> cv2.VideoCapture: Захват следующего кадра с веб-камеры
    cv2.VideoCapture -->> cv2.VideoCapture: Следующий кадр с веб-камеры
    cv2.VideoCapture ->> HandWatchOverlay: Обработка кадра
    HandWatchOverlay ->> cv2.VideoCapture: Захват следующего кадра с веб-камеры
    cv2.VideoCapture -->> cv2.VideoCapture: Следующий кадр с веб-камеры
    cv2.VideoCapture ->> HandWatchOverlay: Обработка кадра
    HandWatchOverlay ->> cv2.VideoCapture: Конец видео, выход из цикла
    cv2.VideoCapture ->> cv2.VideoCapture.release: Освобождение ресурсов видеопотока
    cv2.VideoCapture.release -->> cv2.VideoCapture: Ресурсы освобождены
    cv2.VideoCapture ->> cv2.destroyAllWindows: Закрытие окон
    cv2.destroyAllWindows -->> cv2.VideoCapture: Окна закрыты
    HandWatchOverlay ->> User: Конец программы
```
