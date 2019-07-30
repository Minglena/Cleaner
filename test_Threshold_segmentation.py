
import cv2
import numpy as np

def con_matrix(new_image,weight,height):#拼接矩阵函数，拼接三行三列
    left_matrix = matrix(1, height - 1)
    new_image = np.hstack((left_matrix, new_image))
    new_image = np.hstack((left_matrix, new_image))
    new_image = np.hstack((new_image, left_matrix))
    upper_matrix = matrix(weight + 2, 1)
    new_image = np.vstack((upper_matrix, new_image))
    new_image = np.vstack((upper_matrix, new_image))
    new_image = np.vstack((new_image, upper_matrix))
    return new_image
def matrix(m,n):#创建矩阵函数
    matrix = [None] * n
    for i in range(len(matrix)):
        matrix[i] = [0] * m
    matrix=np.array(matrix)
    matrix=matrix.astype(np.uint8)
    return matrix
    # print(matrix.dtype)
def gradient(image):#梯度幅值
    weight,height=image.shape
    new_image=[]
    for i in range(2,weight-1):
        he=[]
        for j in range(2,height-1):
            fx=(int(image[i,j+1])-int(image[i,j-1]))/2
            fy=(int(image[i+1,j])-int(image[i-1,j]))/2
            f=np.sqrt(fx*fx+fy*fy)
            he.append(f)
            hei=np.array(he)
        new_image.append(hei)
    new_img=np.array(new_image)
    return new_img.astype(np.uint8),height-2,weight-2

src=cv2.imread("C:\\Users\\Minglena\\Pictures\\test\\timg9.jpg")
src=cv2.resize(src,(640,480))
test_gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)#RGB图像转换为HSV图像
test_src=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
H,S,V=cv2.split(test_src)
test_src=cv2.bilateralFilter(test_src,0,30,3)
test_ret1,test_thresh1=cv2.threshold(V,0,255,cv2.THRESH_OTSU)#道路阈值分割

"""
梯度幅值算法，与下面的canny算子只能同时选一个
"""
# test_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 卷积核
# test_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 卷积核
# test_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 卷积核
# t1=cv2.getTickCount()
# new_image,weight,height=gradient(V)
# new_image=con_matrix(new_image,weight,height)
# t2 = cv2.getTickCount()
# t = (t2 - t1) / cv2.getTickFrequency()
# print(t)
# cv2.imshow("new_image",new_image)

"""
canny算子
"""
test_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 卷积核
test_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 卷积核
test_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 卷积核
test_canny=cv2.Canny(test_thresh1,10,150)
new_image = cv2.dilate(test_canny,test_kernel3)  # 膨胀操作
cv2.imshow("canny",new_image)

test_ret2,test_thresh2=cv2.threshold(new_image,0,255,cv2.THRESH_OTSU)
cv2.imshow("test_thresh2",test_thresh2)
add=cv2.add(test_thresh1,test_thresh2)
# cv2.imshow("add",add)
test_add = cv2.dilate(add,test_kernel1)  # 膨胀操作
#面积约束算法
test_erode=cv2.erode(test_add,test_kernel2)
cv2.imshow("erode",test_erode)
cv2.waitKey()