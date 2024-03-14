import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('./balls.jpeg')

# Преобразование изображения в цветовое пространство HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Определение диапазона цветов синего шара в цветовом пространстве HSV
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Создание маски для синих шаров
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Применение морфологических операций для улучшения результата
kernel = np.ones((5, 5), np.uint8)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

# Нахождение контуров синих шаров
contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Отрисовка контуров на исходном изображении
result_image = image.copy()
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# Отображение результата
cv2.imshow('Blue Balls', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
