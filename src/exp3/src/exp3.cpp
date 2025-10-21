#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PI 3.1415926

/***************函数声明，相关参数自行修改***************/
Mat EdgeDetector(Mat input, int threshold);
Mat HoughLines(Mat input, int threshold);
Mat HoughCircles(Mat input, int threshold, int min_radius, int max_radius);

//Mat raw;

int main(int argc, char *argv[])
{
        Mat raw_line = imread("./src/exp3/data/lane.png");
        Mat raw_circle = imread("./src/exp3/data/circle.png");
        Mat raw_edges = imread("./src/exp3/data/edges.png");  
        while (waitKey(10))
        {
                /***************读取图像***************/
                // raw = imread("./src/exp3/data/lane.png");

                if (!raw_line.data || !raw_circle.data || !raw_edges.data)
                {
                        cout << "error" << endl;
                        break;
                }

                imshow("raw_line", raw_line);
                imshow("raw_circle", raw_circle);
                imshow("raw_edges", raw_edges);
                Mat gray_line, gray_circle, gray_edges;
                cvtColor(raw_line, gray_line, COLOR_BGR2GRAY);
                cvtColor(raw_circle, gray_circle, COLOR_BGR2GRAY);
                cvtColor(raw_edges, gray_edges, COLOR_BGR2GRAY);


                /****************调用边缘检测函数****************/
                Mat edge_result =  EdgeDetector(gray_edges, 200);
                imshow("edge_result", edge_result);

                /***************调用霍夫线变换***************/
                Mat lines_result = HoughLines(gray_line, 100);
                imshow("lines_result", lines_result);

                /***************调用霍夫圆变换***************/
                Mat circles_result = HoughCircles(gray_circle, 30, 50, 200);
                imshow("circles_result", circles_result);
        }
        return 0;
}
/***************下面实现EdgeDetector()函数***************/
Mat EdgeDetector(Mat input, int threshold)
{ 
        Mat blured = Mat::zeros(input.size(), CV_64F);
        
        //先高斯滤波
        int T_size = 5;                                    // 模板大小
        Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
        int half_T = T_size / 2;
        double sum = 0.0;
	 double sigma = 1;
        for (int i = 0; i < T_size; i++)
        {
                for (int j = 0; j < T_size; j++)
                {
            		Template.at<double>(i, j) = exp(-(pow(i - half_T, 2) + pow(j - half_T, 2)) / (2 * pow(sigma, 2))) / (2 * PI * pow(sigma, 2));
                	sum += Template.at<double>(i, j); //用于归一化模板元素
                }
        }
        for (int i = 0; i < T_size; i++)
        {
                for (int j = 0; j < T_size; j++)
                {
                Template.at<double>(i, j) /= sum;
                }
        }

        // 卷积
        for (int i = half_T; i < input.rows - half_T; i++)
        {
                for (int j = half_T; j < input.cols - half_T; j++)
                {
                        for (int m = 0; m < T_size; m++)
                        {
                                for (int n = 0; n < T_size; n++)
                                {
                                        blured.at<double>(i, j) += Template.at<double>(m, n) * input.at<uchar>(i + m - half_T, j + n - half_T);
                                }
                        }
                } 
        }
        
    //blured.convertTo(blured, CV_8UC1);
    //imshow("spatial_filtered_image", blured);

        //Sobel算子
        //Mat grad_x = Mat::zeros(input.size(), CV_32S);
        Mat grad_val = Mat::zeros(input.size(), CV_64F);
        Mat grad_angle = Mat::zeros(input.size(), CV_64F);
        Mat Template_x = (Mat_<double>(3,3) << -1,0,1,-2,0,2,-1,0,1);
        Mat Template_y = (Mat_<double>(3,3) << -1,-2,-1,0,0,0,1,2,1);
        T_size = 3;
        half_T = 1;
        
        
        for (int i = half_T; i < input.rows - half_T; i++)
        {
                for (int j = half_T; j < input.cols - half_T; j++)
                {
                        int gx = 0.0;
                        int gy = 0.0;
                        for (int m = 0; m < T_size; m++)
                        {
                                for (int n = 0; n < T_size; n++)
                                {
                                        gx += Template_x.at<double>(m, n) * blured.at<double>(i + m - half_T, j + n - half_T);
                                        gy += Template_y.at<double>(m, n) * blured.at<double>(i + m - half_T, j + n - half_T);
                                }
                        }
                        
                        grad_val.at<double>(i, j) = sqrt(pow(gx, 2) + pow(gy, 2));
                        double theta = atan2(gy, gx) * 180.0 / PI;
                        if (theta < 0) theta += 180.0; // 0-180
                        grad_angle.at<double>(i,j) = theta;
                } 
        }
    	//grad_val.convertTo(grad_val, CV_8UC1);
    	//imshow("grad_val_image", grad_val);
        Mat nms = Mat::zeros(input.size(), CV_64F);
        //非极大值抑制
        for (int i = 1; i < input.rows - 1; i++)
        {
                for (int j = 1; j < input.cols - 1; j++)
                {
                        double theta = grad_angle.at<double>(i, j); 
                        double g = grad_val.at<double>(i, j);
                        double g1, g2;
                        if ((theta > 0 && theta <= 22.5) || (theta > 157.5 && theta <= 180))
                        {
                                g1 = grad_val.at<double>(i, j - 1);
                                g2 = grad_val.at<double>(i, j + 1);
                        }
                        else if (theta > 22.5 && theta <= 67.5)
                        {
                                g1 = grad_val.at<double>(i - 1, j + 1);
                                g2 = grad_val.at<double>(i + 1, j - 1);
                        }
                        else if (theta > 67.5 && theta <= 112.5)
                        {
                                g1 = grad_val.at<double>(i - 1, j);
                                g2 = grad_val.at<double>(i + 1, j);
                        }
                        else if (theta > 112.5 && theta <= 157.5)
                        {
                                g1 = grad_val.at<double>(i - 1, j - 1);
                                g2 = grad_val.at<double>(i + 1, j + 1);
                        }
                        if (g >= g1 && g >= g2)
                        {
                                nms.at<double>(i, j) = g;
                        }
                        else
                        {
                                nms.at<double>(i, j) = 0;
                        }
                }
        }

    	nms.convertTo(nms, CV_8UC1);
    	//imshow("nms_image", nms);
        Mat edge_result = Mat::zeros(input.size(), CV_8U);
        //阈值检测
        for (int i = 0; i < input.rows; i++)
        {
                for (int j = 0; j < input.cols; j++)
                {
                        if (nms.at<uchar>(i, j) > threshold)
                        {
                                edge_result.at<uchar>(i, j) = 255;
                        }

                }
        }

    	//imshow("edge_result_image", edge_result);


        return edge_result;
}
/***************下面实现HoughLines()函数***************/
Mat HoughLines(Mat input, int threshold)
{
        Mat edge = EdgeDetector(input,150);
        
        //离散化参数
        int width = edge.cols;
        int height = edge.rows;
        double diag_len = sqrt(width * width + height * height);
        int rhos = (int)(diag_len * 2); //rho范围[-diag_len, diag_len]
        int thetas = 180;                //theta范围[0, 180)

        Mat accumulator = Mat::zeros(rhos, thetas, CV_32S);

        //投票
        for (int y = 0; y < height; y++)
        {
                for (int x = 0; x < width; x++)
                {
                        if (edge.at<uchar>(y, x) == 255)
                        {
                                for (int t = 0; t < thetas; t++)
                                {
                                        double rho = x * cos(t * PI / 180.0) + y * sin(t * PI / 180.0);
                                        int rho_idx = (int)(rho + diag_len);

                                        if (rho_idx >= 0 && rho_idx < rhos)
                                        {
                                                accumulator.at<int>(rho_idx, t)++;
                                        }
                                }
                        }
                }

        }

        //提取直线
        //在黑白原图中画彩色图像
        cvtColor(edge, edge, COLOR_GRAY2BGR);
        for (int t = 0; t < thetas; t++)
        {
                for (int r = 0; r < rhos; r++)
               {
                        int value = accumulator.at<int>(r, t);
                        if (value > threshold)
                        {
                                //检查是否为局部最大值
                                bool isLocalMax = true;
                                for (int dr = -1; dr <= 1; dr++)
                                {
                                        for (int dt = -1; dt <= 1; dt++)
                                        {
                                                if (t + dt >= 0 && t + dt < thetas && r + dr >= 0 && r + dr < rhos)
                                                {
                                                        if (accumulator.at<int>(r + dr, t + dt) > value)
                                                        {
                                                                isLocalMax = false;
                                                                break;
                                                        }
                                                }
                                        }
                                }

                                if (isLocalMax)
                                { 
                                        double rho = r - diag_len;
                                        double theta = t * PI / 180.0;

                                        //计算直线的两个端点
                                        Point pt1, pt2;
                                        double a = cos(theta);
                                        double b = sin(theta);
                                        double x0 = a * rho;
                                        double y0 = b * rho;
                                        pt1.x = cvRound(x0 + 1000 * (-b));
                                        pt1.y = cvRound(y0 + 1000 * (a));
                                        pt2.x = cvRound(x0 - 1000 * (-b));
                                        pt2.y = cvRound(y0 - 1000 * (a));

                                        //在结果图像上绘制直线
                                        line(edge, pt1, pt2, Scalar(0, 0, 255), 2);
                                }
                        }

               }
 
        }
        return edge;
}
/***************下面实现HoughCircles()函数***************/
Mat HoughCircles(Mat input, int threshold, int min_radius, int max_radius)
{ 
        Mat edge = EdgeDetector(input, 150);      

        // 设置参数范围
        int width = edge.cols;
        int height = edge.rows;
    
        vector<Mat> accumulator(max_radius - min_radius);
        for (int r = 0; r < max_radius - min_radius; r++) {
                accumulator[r] = Mat::zeros(height, width, CV_32S);
        }

        // 投票
        for (int y = 0; y < height; y++) 
        {
                for (int x = 0; x < width; x++) 
                {
                        if (edge.at<uchar>(y, x) == 255) {  // 边缘点
                                // 对每个可能的半径进行投票
                                for (int r = min_radius; r < max_radius; r++) 
                                {
                                        // 对圆周上的点进行投票
                                        for (int angle = 0; angle < 360; angle+=5) 
                                        {  
                                                double theta = angle * PI / 180.0;
                                                int a = (int)(x - r * cos(theta));  // 圆心x坐标
                                                int b = (int)(y - r * sin(theta));  // 圆心y坐标
                                                
                                                // 检查边界
                                                if (a >= 0 && a < width && b >= 0 && b < height) 
                                                {
                                                        accumulator[r - min_radius].at<int>(b, a)++;
                                                }
                                        }
                                }
                        }
                }
        }

        cvtColor(input, input, COLOR_GRAY2BGR);
        Mat result = input.clone();

        // 提取圆
        // 在原图中画彩色图像
        for (int r = 0; r < max_radius - min_radius; r++) 
        { 
                for (int y = 0; y < height; y++) 
                {
                        for (int x = 0; x < width; x++) 
                        {
                                int value = accumulator[r].at<int>(y, x);
                                if (value > threshold) 
                                {
                                        // 检查是否为局部最大值
                                        bool isLocalMax = true;
                                        for (int dy = -1; dy <= 1; dy++) 
                                        {
                                                for (int dx = -1; dx <= 1; dx++) 
                                                {
                                                        if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) 
                                                        {
                                                                if (accumulator[r].at<int>(y + dy, x + dx) > value) 
                                                                {
                                                                        isLocalMax = false;
                                                                        break;
                                                                }
                                                        }
                                                }
                                        }

                                        if (isLocalMax) 
                                        {
                                                // 在结果图像上绘制圆
                                                circle(result, Point(x, y), r + min_radius, Scalar(0, 255, 0), 2);
                                        }
                                }
                        }
                }
        }
        return result;
}
