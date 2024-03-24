import cv2

# Загрузка кадра и изображения PNG
frame = cv2.imread('watch1.png')  # Замените 'watch1.png' на ваш файл изображения
png_image = cv2.imread('watch1.png', cv2.IMREAD_UNCHANGED)

# Проверка, что изображения загружены успешно
if frame is None or png_image is None:
    print("Ошибка: Не удалось загрузить изображения.")
else:
    # Извлечение альфа-канала изображения PNG
    alpha_channel = png_image[:, :, 3]

    # Извлечение RGB-каналов изображения PNG
    png_rgb = png_image[:, :, :3]

    # Определение области, на которую будет наложено изображение PNG
    x, y = 100, 100  # Примерные координаты для наложения изображения PNG
    height, width = png_rgb.shape[:2]

    # Область кадра, на которую будет наложено изображение PNG
    frame_roi = frame[y:y + height, x:x + width]

    # Создание маски на основе альфа-канала
    mask = alpha_channel / 255.0

    # Наложение изображения PNG на кадр с использованием маски
    overlay_area = cv2.multiply(mask[:, :, None], png_rgb.astype(float))
    background_area = cv2.multiply(1.0 - mask[:, :, None], frame_roi.astype(float))
    frame[y:y + height, x:x + width] = cv2.add(overlay_area, background_area).astype('uint8')

    # Отображение результата
    cv2.imshow("Overlay", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
