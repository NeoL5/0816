# -*- coding: utf-8 -*-
# @Author  : Zhiyi Leung
# @Time    : 2024/8/15 15:50
import cv2
import numpy as np
import streamlit as st


# 亮度
def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


# 高斯模糊
def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


# 细节增强
def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def video_frame(original_frame, num_of_blur, num_of_brighten, filter_enhance):
    angle_threshold = 10  # 角度阈值，滤除接近水平的线条
    # 将frame转换为NumPy数组img，转换后包含(height, width, 3)
    img = np.array(original_frame)

    img = brighten_image(img, num_of_brighten)
    img = blur_image(img, num_of_blur)

    if filter_enhance:
        img = enhance_details(img)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行边缘检测
    edges = cv2.Canny(gray_image, 50, 150)
    # 使用霍夫变换检测线条
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # 检查是否检测到线条
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算线条的角度
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

            # 转换为相对于水平面的角度
            if angle < 0:
                angle += 180
            if angle > 90:
                angle = 180 - angle

            # 过滤掉接近水平的线条
            if angle < angle_threshold:
                continue

            # 绘制检测到的线条
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # 将角度标注在图像上
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2  # 计算线段的中点
            text = f'{angle:.1f}'
            cv2.putText(img, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img


def activate_camera(num_of_blur, num_of_brighten, filter_enhance):
    video_placeholder = st.empty()
    # 打开摄像头，参数 0 代表第一个摄像头
    cap = cv2.VideoCapture("http://192.168.137.145:4747/mjpegfeed")

    while True:
        # 读取摄像头画面
        ret, frame = cap.read()

        # 检查是否成功读取
        if not ret:
            print("无法获取摄像头画面")
            break

        # 展示画面
        video_placeholder.image(video_frame(frame, num_of_blur, num_of_brighten, filter_enhance), channels="BGR")

    # 释放摄像头
    cap.release()


def main_loop():
    st.title("OpenCV Application💕")
    st.subheader("This app allows you to play with Image filters!")
    # 侧边栏滑动条
    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    # 侧边栏复选框
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
    activate_camera(blur_rate, brightness_amount, apply_enhancement_filter)


if __name__ == '__main__':
    main_loop()
