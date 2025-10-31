import os
import cv2
import numpy as np
# 对Lena的图片进行边缘提取，使用Sobel, Roberts, Prewitt, LoG, Canny；

def Sobel(img):
    if img is None:
        return None
    kenel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kenel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    # 使用 float32 以避免溢出并让 OpenCV 支持的深度函数可用
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_x = cv2.filter2D(img_gray, cv2.CV_32F, kenel_x)
    img_y = cv2.filter2D(img_gray, cv2.CV_32F, kenel_y)
    img_sobel = np.sqrt(img_x ** 2 + img_y ** 2)
    # 先 normalize 到 [0,255]（不指定不被支持的 dtype），再转为 uint8
    img_sobel = cv2.normalize(img_sobel, None, 0, 255, cv2.NORM_MINMAX)
    img_sobel = img_sobel.astype(np.uint8)
    return img_sobel


def Roberts(img):
    if img is None:
        return None
    kenel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kenel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_x = cv2.filter2D(img_gray, cv2.CV_32F, kenel_x)
    img_y = cv2.filter2D(img_gray, cv2.CV_32F, kenel_y)
    img_roberts = np.sqrt(img_x ** 2 + img_y ** 2)
    img_roberts = cv2.normalize(img_roberts, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_roberts

def Prewitt(img):
    if img is None:
        return None
    kenel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kenel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_x = cv2.filter2D(img_gray, cv2.CV_32F, kenel_x)
    img_y = cv2.filter2D(img_gray, cv2.CV_32F, kenel_y)
    img_prewitt = np.sqrt(img_x ** 2 + img_y ** 2)
    img_prewitt = cv2.normalize(img_prewitt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_prewitt

def LoG(img, sigma=1.0):
    if img is None:
        return None
    size = 5
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (x**2 + y**2 - 2*sigma**2)/(sigma**4)*np.exp(-(x**2 + y**2)/(2*sigma**2))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_log = cv2.filter2D(img_gray, cv2.CV_32F, kernel.astype(np.float32))
    img_log = np.maximum(img_log, 0)
    img_log = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img_log

def Canny(img,threshold1,threshold2):
    if img is None:
        return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, threshold1, threshold2)
    return img_canny

if __name__ == '__main__':
    img_path = "lena.bmp"
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {os.path.abspath(img_path)} ")
    else:
        cv2.imshow('img', img)
    img_sobel = Sobel(img)
    img_roberts = Roberts(img)
    img_prewitt = Prewitt(img)
    img_log = LoG(img)
    img_canny = Canny(img,100,200)
    # 只有在对应结果不为 None 时显示
    if img_sobel is not None:
        cv2.imshow('img_sobel', img_sobel)
    if img_roberts is not None:
        cv2.imshow('img_roberts', img_roberts)
    if img_prewitt is not None:
        cv2.imshow('img_prewitt', img_prewitt)
    if img_log is not None:
        cv2.imshow('img_log', img_log)
    if img_canny is not None:
        cv2.imshow('img_canny', img_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()