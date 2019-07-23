import cv2 as cv
import numpy as np
import sys

def tracker(ret,bbox,frame_copy):#多目标跟踪函数
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[4]
    if tracker_type == 'BOOSTING':
        tracker = cv.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv.TrackerGOTURN_create()
    if not ret:
        print("Cannot read video file")
        sys.exit()
    ret=tracker.init(frame_copy,bbox)
    ret,bbox=tracker.update(frame_copy)
    if ret:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame_copy, p1, p2, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv.putText(frame_copy, "Tracking failure detected", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    return frame_copy

def mat(box):
    mat=-1
    box_mat=[]
    for bbox in box:
        mat=mat+1
        box_mat[mat]=bbox

def mid_point(bbox):#取中心点函数，返回值：（x,y）
    mid_points=[]
    for bbbox in bbox:
        mid_point1 = np.int0(bbbox[0] + bbbox[2] / 2)
        mid_point2 = np.int0(bbbox[1] + bbbox[3] / 2)
        mid_points.append([mid_point1,mid_point2])
    return mid_points

def count_box(box):#处理数字函数，对数字进行排序
    ct=[]
    ti=0
    for bo in box:
        ti=ti+1
        # print(bo)
        if len(bo) is 3:
            ct.append(bo[2])
        else:
            if ti==1:
                ct.append(0)
    # print(type(ct))
    ct=list(set(ct))
    l=len(ct)
    # print(l)
    j=0
    # print(ct)
    if ct:
        ct.sort()
        # print(ct)
        for ia in ct:
            j=j+1
            if ia > j:
                return j
                break
            elif ia<j:
                return ia+1
        if ia==j:
            return j+1

    # ct.sort(reverse=True)
    # print(ct)


    # if ct[1] is j:
    #     return j+1

def mid_sqr(mid):#寻找前后最近的两点，返回值：（x,y,min_sqrt）
    ct=[]
    i=-1
    for sq in mid:
        i=i+3
        if sq[i] or sq[i]==0:
            ct.append(sq[i])
    ct = list(set(ct))
    ct.sort()
    # print(ct)
    for sq in mid:
        return ([sq[0],sq[1],ct[0]])

def sum_sqrt(s_sqrt,sn):
    ct=[]

    sm=[]
    # print(sn)
    for sq in sn:
        if sq:
            ct.append(sq[0])
    # print(ct)
    for sq in s_sqrt:
        ct.append(sq[0])

    ct = list(set(ct))
    print(ct)
    s_sqrt.extend(sn)
    print(s_sqrt)
    for c in ct:
        su = 0
        for sq in s_sqrt:
            if sq[0]==c:
                su=su+sq[1]
        sm.append([c, su])
    return sm

cap=cv.VideoCapture("C:\\Users\\Minglena\\Pictures\\test\\test9.mp4")
pre_lines=[[-335,720,263,432],[1027,720,728,432]]
count=0
cou=True
txy=[]

s=[]
tracker_multi = cv.MultiTracker_create()
while 1:
    count=count+1
    ret,frame=cap.read()#读取视频图像帧
    frame = cv.resize(frame, (640, 360), interpolation=cv.INTER_CUBIC)  # 重设窗口大小

    # 原图中书本的四个角点(左上、右上、左下、右下),与变换后矩阵位置
    # pts1 = np.float32([[100, 0], [540, 0], [0, 360], [640, 360]])
    # pts2 = np.float32([[0, 0], [640, 0], [0, 360], [640, 360]])

    # 生成透视变换矩阵；进行透视变换
    # M1 = cv.getPerspectiveTransform(pts1, pts2)
    # frame = cv.warpPerspective(frame, M1, (640, 360))
    cv.imshow("frame",frame)
    frame_copy=frame.copy()

    if count is 1:
        gray_firstFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 灰度化
        firstFrame = cv.bilateralFilter(gray_firstFrame, 0, 5, 5)  # 双边模糊
        # firstFrame =cv.medianBlur(gray_firstFrame, 15)
        prveFrame = firstFrame.copy()
        background = prveFrame.copy()
    else:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)#灰度化
        # cv.imshow("gray_frame",gray_frame)
        # gray_frame=cv.medianBlur(gray_frame,5)
        gray_frame = cv.bilateralFilter(gray_frame, 0, 30, 3)  # 双边滤波
        # cv.imshow("bliatera",gray_frame)
        # gray_frame = cv.equalizeHist(gray_frame)#直方图均衡化
        # clahe=cv.createCLAHE(clipLimit=1.8,tileGridSize=(4,4))#限制对比度的自适应直方图均衡化
        # gray_frame=clahe.apply(gray_frame)
        # cv.imshow("equalizehist", gray_frame)
        frameDiff = cv.absdiff(prveFrame, gray_frame)#计算两图像之间的差别
        # cv.imshow("frameDiff1", frameDiff)
        r, f = cv.meanStdDev(frameDiff)#求图像均值以及标准差，为下一步的背景图像更新做准备
        # print(r,f)
        if r <0.3:
            background = background
        else:
            background = cv.addWeighted(prveFrame, 0.91, background, 0.31, 0)  # 背景更新算法，超过阈值时进行背景更新,（背景更新为0.1，0.99，前景更新为0.91，0.31）
        cv.imshow("background", background)
        prveFrame = gray_frame.copy()
        frameDiff = cv.subtract(background, gray_frame)  # 图像的减操作
        # cv.imshow("framediff0",frameDiff)
        frameDiff = cv.GaussianBlur(frameDiff, (7, 7), 0)  # 高斯模糊算法
        cv.imshow("frameDiff2", frameDiff)  # 忽略较小的差别
        retVal, thresh = cv.threshold(frameDiff, 63, 255, cv.THRESH_BINARY)  # 二值化，改变255前面的值可以改善背景（背景更新为0.83，前景更新为0.58）
        # thresh =cv.adaptiveThreshold(frameDiff,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,25, 5)
        thresh = cv.dilate(thresh, None, iterations=1)#膨胀操作
        cv.imshow("thresh2",thresh)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # 卷积核
        thresh = cv.dilate(thresh, kernel)  # 膨胀操作
        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓发现，基于二值图像
        text = "Unoccupied"
        # 遍历轮廓
        t = 0
        count_bbox=0
        bbox=[]
        for contour in contours:
            # if contour is too small, just ignore it
            count_bbox=count_bbox+1
            #下面步骤是面积变换
            # print(contour)
            M = cv.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # print(cx,cy)
            S = cv.contourArea(contour)
            S = S * (2 - cy / 360)
            if S < 100:  # 面积阈值
                continue
            elif S > 1000:
                continue
            # 计算最小外接矩形（非旋转）
            (x, y, w, h) = cv.boundingRect(contour)
            # cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if count_bbox:
                bbox.append([x,y,w,h])
            text = "Occupied!"
            # t = t + 1
            # cv.putText(frame, str(t), (x + w % 2, y + h), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # ROI = np.zeros(frame.shape, np.uint8)  # 基于轮廓的ROI区域提取
            cv.drawContours(frame, contour, -1, (0, 200, 500), 2)
            # imgroi = cv.bitwise_and(ROI, frame)
            # cv.imshow("imgroi",imgroi)
            hsv_roi = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mas = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            roi_hist = cv.calcHist([hsv_roi], [0], mas, [180], [0, 180])
            cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
            track_window = (x, y, w, h)
            term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
            dst = cv.calcBackProject([frame], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv.CamShift(dst, track_window, term_crit)  # 目标追踪
            pts = cv.boxPoints(ret)
            pts = np.int0(pts)
            # print(ret)
            # img2 = cv.polylines(frame, [pts], True, 255, 2)
            rect = cv.minAreaRect(contour)  # 最小外接矩形
            box = np.int0(cv.boxPoints(rect))  # 矩形的四个角点取整
            cv.drawContours(frame, [box], 0, (255, 0, 0), 2)

            # pt.push_back(Point(x + w / 2, y + h / 2))
            # for li in range(0,len(ret),1):
            #     cv.line(frame, ret[li], ret[li + 1], cv.Scalar(0, 255, 0), 2.5);
            # cv.imshow("img2", img2)
            # cv.imshow("one_by_one",frame_copy)
            # cv.waitKey(100)
        # frame_copy = tracker(ret, bbox, frame_copy)
        print(bbox)
        if bbox is None:
            break
        if bbox and cou is True:
            bbox1=np.array(bbox)
            # print(bbox1)
            count=2
            cou=False
        elif count is 3:
            bbox2=np.array(bbox)
        elif count is 4:
            bbox3=np.array(bbox)
        else:
            if count is 2:
                count = 1
            else:
                bbox2=bbox3
                # print(bbox2)
                bbox3=np.array(bbox)
                # print(bbox3)
        # print(bbox)
        # print(count)
        if count is 2:#第一帧图像的标记
            mid_bx1=mid_point(bbox1)
            # print(mid_bx1)
            for bx in mid_bx1:

                t = t + 1
                cv.putText(frame, str(t), (bx[0],bx[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # print(t)
                bx.append(t)

                # print(bx)
            # print(mid_bx1)
            # print("bx1")
        t=0
        s_sqrt = []
        l_point=1000 #两点距离阈值
        if count is 3:#第二帧图像的标记
            # mid_bx1 = mid_point(bbox1)
            mid_bx2=mid_point(bbox2)
            print("open")
            # print(mid_bx2)
            # print("off")
            co = True
            # txy.append(np.array(mid_bx1))
            for bx2 in mid_bx2:
                # print(bx2)
                mi_sqr=[]
                mi_sqrt = []
                mid_s=[]
                for bx1 in mid_bx1:

                    sqrt=(bx1[0]-bx2[0])*(bx1[0]-bx2[0])+(bx1[1]-bx2[1])*(bx1[1]-bx2[1])
                    mi_sqr.append(bx2[0])
                    mi_sqr.append(bx2[1])
                    mi_sqr.append(sqrt)
                    mi_sqrt.append(mi_sqr)
                # print(mid_bx2)
                mid_s=mid_sqr(mi_sqrt)
                # print(mid_s[2])
                for bx1 in mid_bx1:
                    # print(bx1)
                    # print(mid_s)
                    sqrt=(bx1[0]-bx2[0])*(bx1[0]-bx2[0])+(bx1[1]-bx2[1])*(bx1[1]-bx2[1])
                    # print(sqrt)
                    # print("bx1")
                    if sqrt<l_point:
                        if mid_s[2]==sqrt:
                            # print("t")
                            t = bx1[2]
                            cv.putText(frame, str(t), (mid_s[0], mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            bx2.append(t)
                            s_sqrt.append([t, sqrt])
                l=len(bx2)
                if l==2:
                    t=count_box(mid_bx2)
                    cv.putText(frame, str(t), (mid_s[0], mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    bx2.append(t)
            if mid_bx2:
            #     txy.append((np.array(mid_bx2)).astype(np.float64))
                txy.append(mid_bx2)
                # print(bx2)
            # print(mid_bx2)
            #     print(mid_s)
        else:
            # print(count)
            if count is 1:
                count=2
            elif count is 2:
                count=2
            else:
                # print("open")
                # mid_bx2 = mid_point(bbox2)
                mid_bx3=mid_point(bbox3)
                # print(mid_bx2)
                # print(mid_bx3)
                # print("off")
                co = True

                for bx3 in mid_bx3:

                    mi_sqr = []
                    mi_sqrt = []
                    mid_s = []
                    for bx2 in mid_bx2:#将第二和第三帧图像各点的距离求出
                        # print(mid_bx2)
                        sqrt = (bx2[0] - bx3[0]) * (bx2[0] - bx3[0]) + (bx2[1] - bx3[1]) * (bx2[1] - bx3[1])
                        mi_sqr.append(bx3[0])
                        mi_sqr.append(bx3[1])
                        mi_sqr.append(sqrt)
                        mi_sqrt.append(mi_sqr)
                    co = False
                    # print(mid_bx2)
                    mid_s = mid_sqr(mi_sqrt)
                    # print(mid_s)
                    for bx2 in mid_bx2:#将图像标记并记录其标号
                        sqrt = (bx2[0] - bx3[0]) * (bx2[0] - bx3[0]) + (bx2[1] - bx3[1]) * (bx2[1] - bx3[1])
                        # print(sqrt)
                        if sqrt<l_point or sqrt==0:
                            if mid_s[2]==sqrt:
                                t=bx2[2]
                                cv.putText(frame, str(t), (mid_s[0], mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv.putText(frame,"+", (mid_s[0]+40, mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                                for ss in s:
                                    if t==ss[0]:
                                        cv.putText(frame,str(ss[1]), (mid_s[0] + 80, mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                                bx3.append(t)
                                s_sqrt.append([t,sqrt])
                    l = len(bx3)
                    if l == 2:
                        if mid_s is None or mid_bx3 is None:
                            pass
                        else:
                            t = count_box(mid_bx3)
                            cv.putText(frame, str(t), (mid_s[0], mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv.putText(frame, "+", (mid_s[0] + 40, mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            for ss in s:
                                if t == ss[0]:
                                    cv.putText(frame, str(ss[1]), (mid_s[0] + 80, mid_s[1]), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                            bx3.append(t)
                # print(mid_bx2)
                mid_bx2=mid_bx3
                # print(mid_bx2)
                # print(s)
                print(s_sqrt)
                sm=sum_sqrt(s_sqrt,s)
                s=sm
                print(s)
                # mid_bx2_array=np.array(mid_bx2)
                # print(mid_bx2_array)
                if mid_bx2:
                #     txy.append((np.array(mid_bx2)).astype(np.float64))
                    txy.append(mid_bx2)
                # print(txy)
                    # print(mid_s)
                    # print(bx2)
                    # print(mid_bx2)
                    # print(mi_sqrt)

        print("stop")
        cv.putText(frame, "F{}".format(text), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('frame_with_result', frame)
        # print(bbox)
        # print(count)
        # for bbbox in bbox:

        #多目标跟踪函数
        # count_box=0
        # boxes=[]
        # for bbbox in bbox:
        #     count_box=count_box+1
        #     bbbox=np.array(bbbox)
        #     bbboxes=bbbox.astype(np.float64)
        #     if count_box is 1:
        #         boxes = bbboxes
        #     else:
        #         boxes=np.vstack((boxes,bbboxes))#将两个数组拼接在一起
        #
        #     # print(bbbox)
        #     # print(bbboxes)
        #     print(type(bbbox))
        #     # bbbox=bbbox.astype(np.float64)
        # # print(boxes.dtype)
        # print(type(boxes))
        # if count%5==0 or count==2:
        #     boxes=tuple(boxes)
        #     ret = tracker_multi.add(cv.TrackerMIL_create(), frame_copy, boxes)
        #     # for bbbox in boxes:
        #     #     bbbox=bbbox.tolist()
        #     #     tuple(bbbox)
        #     #     print(bbbox)
        #     #     ret=tracker_multi.add(cv.TrackerMIL_create(), frame_copy, bbbox)
        #
        # ret,boxes = tracker_multi.update(frame_copy)
        # # print(ret,boxes)
        # for newbox in boxes:
        #     print(newbox)
        #     p1 = (int(newbox[0]), int(newbox[1]))
        #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        #     cv.rectangle(frame_copy, p1, p2, (200, 0, 0))
        # cv.imshow("frame_copy",frame_copy)
        # print('stop')

        # cv.imshow("one_by_one", frame_copy)
        # cv.imshow('thresh', thresh)
        # cv.imshow('frameDiff', frameDiff)
        # 处理按键效果
        key = cv.waitKey(1) & 0xff  # 延迟时间
        if key == 27:  # 按下ESC时，退出
            break
        elif key == ord(' '):  # 按下空格键时，暂停
            cv.waitKey(0)
    cv.waitKey(10)