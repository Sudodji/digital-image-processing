import cv2

# Загрузка изображения
image = cv2.imread('./lenna.png', cv2.IMREAD_GRAYSCALE)

# Применение гистограммной эквализации
equalized_image = cv2.equalizeHist(image)

# Сохранение эквализованного изображения
cv2.imwrite('equalized_image.jpg', equalized_image)
