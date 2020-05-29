# import numpy as np
import cv2

# 人脸识别分类器
faceCascade = cv2.CascadeClassifier(
    './cascades/haarcascade_frontalface_default.xml')

# 识别眼睛的分类器
eyeCascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')

# 开启摄像头
cap = cv2.VideoCapture(1)

# 设定写入参数
fps = 30
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vout_1 = cv2.VideoWriter()
vout_1.open('./output/output.mp4', fourcc, fps, size, True)

flage = True
while flage:
    # 读取摄像头中的图像，ok为是否读取成功的判断参数
    flage, img = cap.read()
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(32, 32)
    )
    result = []
    # 在检测人脸的基础上检测眼睛
    for (x, y, w, h) in faces:
        fac_gray = gray[y: (y+h), x: (x+w)]
        eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 2)

        # 眼睛坐标的换算，将相对位置换成绝对位置
        for (ex, ey, ew, eh) in eyes:
            result.append((x+ex, y+ey, ew, eh))

    # 画矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (ex, ey, ew, eh) in result:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    vout_1.write(img)
    cv2.imshow('video', img)
    k = cv2.waitKey(1)
    if k == 27:    # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
