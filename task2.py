import numpy as np
import cv2

def histogram_equalization(image):
    # Преобразование изображения в 8-битный формат (если оно не такое)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Получение гистограммы изображения
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0,256])

    # Вычисление кумулятивной функции распределения
    cdf = histogram.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Нормализация кумулятивной функции

    # Применение эквализации
    equalized_image = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape)

    # Приведение значений обратно к 8-битному диапазону
    equalized_image = equalized_image.astype(np.uint8)

    return equalized_image


# Загрузка изображения
image = cv2.imread('./images/lenna.png', cv2.IMREAD_GRAYSCALE)

# Применение гистограммной эквализации
equalized_image = histogram_equalization(image)

# Сохранение эквализованного изображения
cv2.imwrite('equalized_image.jpg', equalized_image)
