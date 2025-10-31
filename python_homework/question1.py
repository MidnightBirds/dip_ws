import cv2
import numpy as np
#设计LCD屏幕坏点检测算法

def detect_bad_points(img):
    if img is None: 
        return None
    kenel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_sharp = cv2.filter2D(img_gray, -1, kenel)

    _, binary_img = cv2.threshold(img_sharp, 100, 255, cv2.THRESH_BINARY)

    return binary_img  


if __name__ == '__main__':
    img = cv2.imread('lcd.jpg')
    cv2.imshow('img', img)
    img_sharp = detect_bad_points(img)
    cv2.imshow('img_sharp', img_sharp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()