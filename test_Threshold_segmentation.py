import cv2
import numpy as np

def gradient(image):#梯度幅值
    weight,height=image.shape
    new_image=[]
    for i in range(2,weight-1):
        he=[]
        # if i==2:
        #     for j in range(1,height):
        #         he.append(0)
        #
        for j in range(2,height-1):
            fx=(int(image[i,j+1])-int(image[i,j-1]))/2
            fy=(int(image[i+1,j])-int(image[i-1,j]))/2
            f=np.sqrt(fx*fx+fy*fy)
            he.append(f)
            hei=np.array(he)
        new_image.append(hei)
    new_img=np.array(new_image)
    return new_img.astype(np.uint8)


src=cv2.imread("C:\\Users\\Minglena\\Pictures\\test\\timg9.jpg")
src=cv2.resize(src,(640,480))
test_gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
test_src=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
H,S,V=cv2.split(test_src)
cv2.imshow("H",H)
cv2.imshow("S",S)
cv2.imshow("V",V)
# print(V)
test_src=cv2.bilateralFilter(test_src,0,30,3)
test_ret1,test_thresh1=cv2.threshold(V,0,255,cv2.THRESH_OTSU)
cv2.imshow("test_thresh1",test_thresh1)
t1=cv2.getTickCount()
new_image=gradient(V)
t2=cv2.getTickCount()
t=(t2-t1)/cv2.getTickFrequency()
print(t)
cv2.imshow("new_image",new_image)
test_ret2,test_thresh2=cv2.threshold(new_image,0,255,cv2.THRESH_OTSU)
cv2.imshow("test_thresh2",test_thresh2)
test_canny=cv2.Canny(test_thresh1,10,150)
cv2.imshow("canny",test_canny)
# add=cv2.add(thresh1,thresh2)
# # cv2.imshow("add",add)
test_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 卷积核
test_thresh = cv2.dilate(test_thresh1,test_kernel1)  # 膨胀操作
#面积约束算法
test_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 卷积核
test_erode=cv2.erode(test_thresh,test_kernel2)
cv2.imshow("erode",test_erode)
cv2.waitKey()