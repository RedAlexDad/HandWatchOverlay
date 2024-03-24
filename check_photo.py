import cv2


def image_info(image_path):
    print('-' * 50)
    # Загрузка изображения
    img = cv2.imread(image_path)

    if img is None:
        print("Не удалось загрузить изображение. Пожалуйста, убедитесь, что путь к изображению корректен.")
        return

    # Вывод основной информации об изображении
    print("Информация об изображении:", image_path)
    print("Размеры изображения:", img.shape)
    print("Тип данных:", img.dtype)

    # Вычисление и вывод других свойств изображения
    print("Минимальное значение пикселя:", img.min())
    print("Максимальное значение пикселя:", img.max())
    print("Среднее значение пикселя:", img.mean())
    print("Стандартное отклонение значений пикселей:", img.std())

    # Отображение изображения
    # cv2.imshow("Изображение", images)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print('-'*50)


# Пример использования функции
image_info("watch1.png")
image_info("watch2.png")
