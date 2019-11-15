import cv2 as cv
import numpy as np
import cv2
def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]
    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 is None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    # print(x,y)
    return [x, y]
def make_coordinates(image,line_parameters):
    print("line_parameters is",line_parameters)
    print("line_parameters' type is",type(line_parameters))
    # print(np.isnan(line_parameters))
    # line_parameters为None时，若赋给多个值，则会出现cannot unpack non-iterable的报错
    # if line_parameters is None:
    #     pass
    # else:
    #     if lines_judge is True:
    #         last_line_parameters=line_parameters
    #         lines_judge=False
    # if line_parameters is None:
    #     line_parameters=last_line_parameters
    slope,intercept=line_parameters
    # if line_parameters is None:
    #     pass
    # else:
    #     last_line_parameters=line_parameters
    y1=image.shape[0]
    y2=int(y1*(1/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
def bool_judge(left_fit_average,last_left_fit_average,right_fit_average,last_right_fit_average):#暂时没有用到，用于判断是否为布尔类型
    if type(np.isnan(left_fit_average)) is bool:
        left_fit_average = last_left_fit_average
        print("left_fit继承")
    else:
        last_left_fit_average = left_fit_average
        print("last_left_fit继承")
    if type(np.isnan(right_fit_average)) is bool:
        right_fit_average = last_right_fit_average
        print("right-fit继承")
    else:
        last_right_fit_average = right_fit_average
        print("last_right_fit继承")
def average_slope_intercept(image,lines,lines_judge,last_left_fit_average,last_right_fit_average):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)#重新构造数组
        parameters=np.polyfit((x1,x2),(y1,y2),1)#多项式拟合
        # print(parameters)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    # print(left_fit)
    # print(right_fit)
    # if right_fit is None:
    #     right_fit=[(0.9629629629629606, -269.9629629629607), (0.9615384615384606, -267.4230769230757)]
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    if lines_judge is True:
        if np.isnan(left_fit_average) is True:
            pass
        else:
            last_left_fit_average=left_fit_average
            print(last_left_fit_average)
            print("第一帧")
        if np.isnan(right_fit_average) is True:
            pass
        else:
            last_right_fit_average=right_fit_average
            print(last_right_fit_average)
            print("第一帧")
    if lines_judge is False:
        print("left_fit_average is",np.isnan(left_fit_average),type(np.isnan(left_fit_average)))
        print("right_fit_average is",np.isnan(right_fit_average),type(np.isnan(right_fit_average)))
        #np.isnan(left_fit_average)为Ture时，其值的类型为np.bool，而不是bool
        if type(np.isnan(left_fit_average)) is np.ndarray:
            last_left_fit_average = left_fit_average
            print("last_left_fit继承")
        else:
            left_fit_average = last_left_fit_average
            print("left_fit继承")
        if type(np.isnan(right_fit_average)) is np.ndarray:
            last_right_fit_average = right_fit_average
            print("last_right_fit继承")
        else:
            right_fit_average = last_right_fit_average
            print("right-fit继承")
    print("left_fit_average's type is",type(left_fit_average))
    left_line=make_coordinates(image,left_fit_average)
    right_line=make_coordinates(image,right_fit_average)
    # print(left_line)
    # print(right_line)
    x,y=cross_point(left_line,right_line)
    left_line[2]=x
    left_line[3]=y
    right_line[2]=x
    right_line[3]=y
    return np.array([left_line,right_line]),last_left_fit_average,last_right_fit_average
def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
    canny = cv.Canny(blur, 50, 100)  # canny算子进行边缘检测，deep，high用于筛选线条的数量（长度？）
    return canny
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv.line(line_image,(x1,y1),(x2,y2),(0,255,0),1)
    return line_image
def region_of_interest(image):
    height=image.shape[0]
    polygons=np.array([[(-300,height),(500,height),(290,140)]])
    mask=np.zeros_like(image)
    cv.fillPoly(mask,polygons,255)
    masked_image=cv.bitwise_and(image,mask)
    # cv.imshow("mask",mask)
    return masked_image
def fill_color_demo(image,m,n):#泛洪填充
    h,w,ch=image.shape[:]#求图片的宽高以及通道
    # print(h)
    # print(w)
    # print(ch)
    mask=np.zeros([h+2,w+2],np.uint8)#创建图像数组
    cv.floodFill(image,mask,(3+m,3+n),(255,255,255),(60,60,60),(10,10,10),cv.FLOODFILL_FIXED_RANGE)
    cv.floodFill(image, mask, (640-m,3+n), (255, 255, 255), (60, 60, 60), (10, 10, 10), cv.FLOODFILL_FIXED_RANGE)
    return image
# def left_right(averaged):
#     if averaged[1]

# image=cv.imread("C:\\Users\\Minglena\\Pictures\\test\\timg3.jpg")
# lane_image=np.copy(image)
# canny=canny(lane_image)
# cropped_image=region_of_interest(canny)
# cv.imshow("cropped_image",cropped_image)
# lines=cv.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
# averaged_lines=average_slope_intercept(lane_image,lines)
# line_image=display_lines(lane_image,averaged_lines)
# combo_image=cv.addWeighted(lane_image,0.8,line_image,1,1)
# cv.imshow("result",combo_image)
# cv.waitKey(0)
cap=cv.VideoCapture("C:\\Users\\Minglena\\Pictures\\test\\test3.mp4")
pre_lines=[[-335,720,263,432],[1027,720,728,432]]
count=0
lines_judge=True
last_left_fit_average=[]
last_right_fit_average=[]
while 1:
    count=count+1
    ret,frame=cap.read()
    frame=cv.resize(frame,(640,360),interpolation=cv.INTER_CUBIC)
    cv.imshow("frame",frame)
    cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    frame=cv.bilateralFilter(frame, 0, 30, 3)
    canny_image = canny(frame)
    cv.imshow("canny_image",canny_image)
    # cropped_image = region_of_interest(canny_image)
    # cv.imshow("cropped_image",cropped_image)
    lines = cv.HoughLinesP(canny_image, 1, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    if lines is None:#用于继承上一帧数据
        lines = pre_lines
        lines = np.array(lines)
        lines = lines.astype(np.float64)
        print("空")
    pre_lines=lines
    averaged_lines,last_left_fit_average,last_right_fit_average = average_slope_intercept(frame, lines,lines_judge,last_left_fit_average,last_right_fit_average)
    lines_judge=False
    print(averaged_lines)
    line_image = display_lines(frame, averaged_lines)
    fill_color_demo(line_image,1,1)
    gray=cv.cvtColor(line_image,cv.COLOR_BGR2GRAY)
    re,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    contours,hierarchy= cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    height = frame.shape[0]
    wight = frame.shape[1]
    mask = np.zeros((height, wight, 3), dtype='uint8')
    cv.drawContours(mask, contours, -1, (0, 200, 500), 2)
    combo_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    cv.imshow("result", combo_image)
    if count is 1:
        firstFrame = cv.resize(combo_image, (640, 360), interpolation=cv.INTER_CUBIC)
        gray_firstFrame = cv.cvtColor(firstFrame, cv.COLOR_BGR2GRAY)  # 灰度化
        # firstFrame = cv2.GaussianBlur(gray_firstFrame, (21, 21), 0)  # 高斯模糊，用于去噪
        firstFrame = cv.bilateralFilter(gray_firstFrame, 0, 50, 5)  # 双边模糊
        prveFrame = firstFrame.copy()
        background = prveFrame.copy()
    else:
        frame = cv2.resize(combo_image, (640, 360), interpolation=cv2.INTER_CUBIC)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        # gray_frame=cv2.blur(gray_frame,(5,1))
        gray_frame = cv2.bilateralFilter(gray_frame, 0, 5, 3)  # 双边滤波
        gray_frame = cv2.equalizeHist(gray_frame)
        # cv2.imshow("current_frame", gray_frame)
        # 计算当前帧与上一帧的差别
        frameDiff = cv2.absdiff(prveFrame, gray_frame)
        # cv2.imshow("frameDiff1", frameDiff)
        # min_val=cv2.minMaxLoc(gray_frame)
        # print(min_val)
        # print(max_val)
        r, f = cv2.meanStdDev(frameDiff)
        # print(r)
        if r > 0.8:
            background = prveFrame.copy()
        else:
            background = cv2.addWeighted(prveFrame, 0.1, background,0.99, 0)
        # Diff=cv2.subtract(prveFrame,gray_frame)
        # background=cv.GaussianBlur(background,(9,9),0)
        # cv2.imshow("background", background)
        prveFrame = gray_frame.copy()
        M1, dev = cv2.threshold(frameDiff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # print(dev)
        frameDiff = cv2.subtract(background, gray_frame)
        frameDiff=cv.GaussianBlur(frameDiff,(15,15),0)
        # cv2.imshow("frameDiff2", frameDiff)
        # 忽略较小的差别
        t = 0
        retVal, thresh = cv2.threshold(frameDiff, 50, 255, cv2.THRESH_BINARY)  # 直方图均衡，改变255前面的值可以改善背景
        # retVal, thresh = cv2.threshold(frameDiff, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 对阈值图像进行填充补洞
        # kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))#卷积核
        # thresh = cv2.dilate(thresh, kernel)#膨胀操作
        # cv.imshow("thresh1",thresh)
        thresh = cv2.dilate(thresh, None, iterations=1)
        # thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)#开操作
        # thresh=cv2.erode(thresh,kernel)#腐蚀操作
        # labels = measure.label(thresh, connectivity=2)  # 8连通区域标记
        # dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
        # print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        # ax1.imshow(thresh, plt.cm.gray, interpolation='nearest')
        # ax1.axis('off')
        # ax2.imshow(dst, interpolation='nearest')
        # ax2.axis('off')
        # fig.tight_layout()
        # plt.show()
        # reagine=skimage.measure.regionprops(thresh)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours[0]))
        # cv2.drawContours(frame,contours,-1,(0,200,500),2)
        # 轮廓发现，基于二值图像
        height = frame.shape[0]
        wight = frame.shape[1]
        mask = np.zeros((height, wight, 3), dtype='uint8')
        cv2.drawContours(mask, contours, -1, (0, 200, 500), 2)
        # cv2.imshow("mask", mask)
        text = "Unoccupied"

        # 遍历轮廓
        for contour in contours:
            # if contour is too small, just ignore it

            if cv2.contourArea(contour)< 50:  # 面积阈值

                continue
            elif cv2.contourArea(contour)>150:

                continue
            # 计算最小外接矩形（非旋转）

            (x, y, w, h) = cv2.boundingRect(contour)
            # term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            # re, track_window = cv2.meanShift(thresh, track_window, term_crit)
            # a, b, wi, he = track_window
            # img2 = cv2.rectangle(frame, (a, b), (a + wi, b + he), 255, 2)
            # cv2.imshow("camshift", img2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Occupied!"
            ROI = np.zeros(frame.shape, np.uint8)  # 基于轮廓的ROI区域提取
            # cv2.drawContours(ROI, contours, 1, (255, 255, 255), -1)
            imgroi = cv2.bitwise_and(ROI, frame)
            # cv2.imshow("imgroi",imgroi)
            hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mas = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roi_hist = cv2.calcHist([hsv_roi], [0], mas, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            track_window = (x, y, w, h)
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            dst = cv2.calcBackProject([frame], [0], roi_hist, [0, 180], 1)

            ret, track_window = cv2.CamShift(dst, track_window, term_crit)  # 目标追踪
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            # print(ret)
            # img2 = cv2.polylines(frame, [pts], True, 255, 2)
            # pt.push_back(Point(x + w / 2, y + h / 2))
            # for li in range(0,ret.len(),1):
            #     cv2.line(frame, ret[li], ret[li + 1], cv2.Scalar(0, 255, 0), 2.5);
            # cv2.imshow("img2", img2)
            t = t + 1
            # cv2.putText(frame, str(t), (x + w % 2, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # print(b)
        # cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # cv2.CamShift(thresh,track_window,term_crit)
        cv2.putText(frame, "F{}".format(text), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('frame_with_result', frame)

        # cv2.imshow('thresh', thresh)

        # cv2.imshow('frameDiff', frameDiff)
        # 处理按键效果
        key = cv2.waitKey(100) & 0xff  # 延迟时间
        if key == 27:  # 按下ESC时，退出
            break
        elif key == ord(' '):  # 按下空格键时，暂停
            cv2.waitKey(0)
    cv.waitKey(1)
