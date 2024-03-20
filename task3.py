import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('./wolf.jpg')

# Получение размеров изображения
height, width = image.shape[:2]

# Определение координат углов исходного изображения (параллелограмма)
pts_src = np.array([[126, 0 ], [width, 0], [0, height], [385, height]], dtype=np.float32)

# Задание координат углов целевого прямоугольника
pts_dst = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)

# Получение матрицы преобразования
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Применение перспективного преобразования
rectified_image = cv2.warpPerspective(image, matrix, (width, height))

# Применение фильтра Гаусса для удаления шумов
denoised_image = cv2.GaussianBlur(rectified_image, (5, 5), 0)

# Увеличение резкости с помощью фильтра Лапласиана
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
sharpened_image = cv2.filter2D(denoised_image, -1, sharpen_kernel)

# Отображение изображений
cv2.imshow('Sharpened Image', sharpened_image)
cv2.imwrite('sharpened_image.jpg', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
