import cv2
import numpy as np

def con_matrix(new_image,weight,height,m,n):#拼接矩阵函数，拼接n行m列
    left_matrix = matrix(int(m/2)+1, height - 1)
    right_matrix=matrix(int(m/2),height-1)
    new_image = np.hstack((left_matrix, new_image))
    new_image = np.hstack((new_image, right_matrix))
    upper_matrix = matrix(weight + 2, int(n/2)+1)
    down_matrix=matrix(weight+2,int(n/2))
    new_image = np.vstack((upper_matrix, new_image))
    new_image = np.vstack((new_image, down_matrix))
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
def edge_point_extraction(test_thresh2,G):
    j=0.8
    k=25
    height ,weight= test_thresh2.shape
    print(V.shape)
    print(height,weight)
    new_image = []
    for y in range(1, height):
        he = []
        for x in range(k+1, weight-k-1):
            # print(x)
            if test_thresh2[y,x]==0:
                he.append(0)
            else:
                if x<=weight/2:
                    leftroad=0
                    imagebackgound=0
                    for i in range(0,k):
                        leftroad=leftroad+G[y,x+i]
                        imagebackgound=imagebackgound+G[y,x-i]
                        imagebackgound=imagebackgound+G[y,x+i]
                        road=leftroad
                else:
                    rightroad=0
                    imagebackgound=0
                    for i in range(0,k):
                        rightroad=rightroad+G[y,x-i]
                        imagebackgound=imagebackgound+G[y,x+i]
                        imagebackgound=imagebackgound+G[y,x-i]
                        road=rightroad
                if (1-j)*imagebackgound>road:
                    he.append(255)
                else:
                    he.append(0)
            hei = np.array(he)
        new_image.append(hei)
    new_img = np.array(new_image)
    return new_img.astype(np.uint8), height, weight
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for li in lines:
            for x1,y1,x2,y2 in li:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),1)
    return line_image

img=cv2.imread("C:\\Users\\Minglena\\Pictures\\test\\timg12.jpg")
img=cv2.resize(img,(640,480))
src=cv2.medianBlur(img,9)
test_gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
test_src=cv2.cvtColor(src,cv2.COLOR_BGR2HSV)#RGB图像转换为HSV图像
H,S,V=cv2.split(test_src)
cv2.imshow("V",V)
test_src=cv2.bilateralFilter(test_src,0,30,3)
test_ret1,test_thresh1=cv2.threshold(V,0,255,cv2.THRESH_OTSU)#道路阈值分割
cv2.imshow("test_thresh1",test_thresh1)
test_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 卷积核
test_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))  # 卷积核
test_kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 卷积核
"""
梯度幅值算法，与下面的canny算子只能同时选一个
"""
t1=cv2.getTickCount()
new_image,weight,height=gradient(V)
new_image=con_matrix(new_image,weight,height,3,3)
t2 = cv2.getTickCount()
t = (t2 - t1) / cv2.getTickFrequency()
# print(t)
# cv2.imshow("new_image",new_image)

"""
canny算子
"""
# test_canny=cv2.Canny(test_thresh1,10,150)
# cv2.imshow("test_canny",test_canny)
# print(test_canny)
# new_image = cv2.dilate(test_canny,test_kernel3)  # 膨胀操作
# cv2.imshow("canny",new_image)

test_ret2,test_thresh2=cv2.threshold(new_image,0,255,cv2.THRESH_OTSU)
if test_thresh1[320,240]!=test_thresh2[320,240]:
    for i in range(1,479):
        for j in range(1,639):
            test_thresh1[i,j]=255-test_thresh1[i,j]
cv2.imshow("test_thresh2",test_thresh2)
"""
若中心像素点为黑色，则黑白转换
"""

add=cv2.add(test_thresh1,test_thresh2)
cv2.imshow("add",add)
test_add = cv2.dilate(add,test_kernel1)  # 膨胀操作
#面积约束算法
test_erode=cv2.erode(test_add,test_kernel2)
cv2.imshow("erode",test_erode)
edge_point,we,he=edge_point_extraction(test_thresh2,add)
cv2.imshow("edge_point",edge_point)
lines=cv2.HoughLinesP(edge_point,1, np.pi / 180, 50, np.array([]), minLineLength=20, maxLineGap=20)
print(lines)
line_image=display_lines(img,lines)
cv2.imshow("line_image",line_image)
cv2.waitKey()
"""
8月26日，对梯度幅值进行测试，为进一步的道路边缘提取做准备
"""