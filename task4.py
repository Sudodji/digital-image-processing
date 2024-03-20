import cv2 as cv

img = cv.imread('./images/hieroglyphs.jpeg')
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, img = cv.threshold(img, 80, 255, 0)
cv.imshow('Processed Image', img) # выводим итоговое изображение в окно
cv.imwrite('processed_image.jpg', img)
cv.waitKey()
cv.destroyAllWindows()
