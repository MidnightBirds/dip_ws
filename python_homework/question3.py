import cv2
import numpy as np
import math

def hough_circle_detection_gradient(image, min_radius=10, max_radius=100, threshold=50):

    # 1. 边缘检测（使用Canny）
    edges = cv2.Canny(image, 50, 150)
    
    # 2. 计算梯度（Sobel算子）
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. 创建圆心累加器（二维数组：[x, y]）
    height, width = image.shape[:2]
    center_accumulator = np.zeros((height, width), dtype=np.int32)
    
    # 4. 遍历边缘点，沿梯度方向投票（寻找圆心）
    for y in range(height):
        for x in range(width):
            if edges[y, x] != 0:  # 如果是边缘点
                # 计算梯度方向
                angle = math.atan2(grad_y[y, x], grad_x[y, x])
                
                # 对于每个可能的半径，计算可能的圆心位置
                for r in range(min_radius, max_radius + 1):
                    # 沿着梯度方向（垂直于边缘）计算圆心位置
                    # 圆心在边缘点的梯度方向上
                    center_x = int(x - r * math.cos(angle))
                    center_y = int(y - r * math.sin(angle))
                    
                    # 检查坐标是否在图像范围内
                    if 0 <= center_x < width and 0 <= center_y < height:
                        center_accumulator[center_y, center_x] += 1

    # normalize 返回的可能是非 uint8 类型（例如 CV_32S），imshow 不接受 CV_32S 等深度
    # 先转换为 float 再归一化到 0-255，最后转为 uint8 用于显示
    center_accumulator_disp = cv2.normalize(center_accumulator.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('Center Accumulator', center_accumulator_disp)
    # 5. 寻找圆心位置（累加器中的峰值）
    # 向量化寻找候选并限制数量，避免在低阈值下计算量爆炸
    ys, xs = np.where(center_accumulator >= threshold)
    if len(xs) == 0:
        return []
    votes = center_accumulator[ys, xs]
    # 组合并按投票数降序排序，取 top_k 候选以限制复杂度
    top_k = 50
    idx_sorted = np.argsort(votes)[::-1]
    idx_sorted = idx_sorted[:top_k]
    center_candidates = [(int(xs[i]), int(ys[i]), int(votes[i])) for i in idx_sorted]
    
    # 6. 对每个候选圆心，计算最佳半径
    circles = []
    # 预计算边缘点坐标向量，向量化距离计算
    edge_pts = np.column_stack(np.where(edges != 0))  # [[y,x], ...]
    if edge_pts.size == 0:
        return []

    for (cx, cy, count) in center_candidates:
        # 计算所有边缘点到候选圆心的距离（向量化）
        dy = edge_pts[:, 0].astype(np.float32) - float(cy)
        dx = edge_pts[:, 1].astype(np.float32) - float(cx)
        dists = np.hypot(dx, dy)

        # 把距离四舍五入到最近整数并筛选半径范围
        d_round = np.rint(dists).astype(np.int32)
        mask = (d_round >= min_radius) & (d_round <= max_radius)
        if not np.any(mask):
            continue
        d_sel = d_round[mask]

        # 使用 bincount 统计最常见半径（高效且不会出现空直方图问题）
        counts = np.bincount(d_sel - min_radius, minlength=(max_radius - min_radius + 1))
        best_idx = np.argmax(counts)
        radius = int(best_idx + min_radius)

        # 验证并添加
        if min_radius <= radius <= max_radius and counts[best_idx] > 0:
            circles.append([cx, cy, radius])
    
    return circles


    

if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('circle.png')
    if image is None:
        print("错误：无法读取图像，请确保图像文件存在")
        exit()
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 执行霍夫圆检测（霍夫梯度法）
    circles = hough_circle_detection_gradient(gray, min_radius=10, max_radius=100, threshold=105)
    
    # 绘制检测到的圆
    result = image.copy()
    for (x, y, r) in circles:
        # 绘制圆
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        # 绘制圆心
        cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
    
    # 显示结果
    cv2.imshow('Detected Circles', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()