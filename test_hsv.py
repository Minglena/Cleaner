import cv2

src=cv2.imread("C:\\Users\\Minglena\\Pictures\\test\\joey.jpeg")
img=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
cv2.imshow("img",img)
cv2.waitKey(0)