import cv2

img=cv2.imread("C:\\Users\\Minglena\\Pictures\\test\\square.jpg")
# src=cv2.resize(img,(640,360))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
bilater=cv2.bilateralFilter(gray,0,30,3)
equalize=cv2.equalizeHist(bilater)
cv2.imshow("equalize",equalize)
cv2.waitKey(0)
