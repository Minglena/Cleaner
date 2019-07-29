import pywt
import numpy as np
import matplotlib.pyplot as plt
import cv2

def haar(img):
    coefs = pywt.wavedec2(img, 'haar', level=2)
    coefs[1] = pywt.threshold(coefs[1], 5, 'soft', 0)
    coefs[2] = pywt.threshold(coefs[2], 5, 'soft', 0)
    print(coefs[1], coefs[2])
    con_img = pywt.waverec2(coefs, 'haar')
    con_img = con_img.astype(np.uint8)  # 进行类型转换
    return con_img

img=cv2.imread("C:\\Users\\Minglena\\Pictures\\test\\timg9.jpg")
img=cv2.resize(img,(640,480))
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("img",img)
src=cv2.bilateralFilter(img, 0, 30, 3)  # 双边滤波
cv2.imshow("bilateralFilter",src)
cv2.imshow("haar",haar(img))
cv2.waitKey(0)
