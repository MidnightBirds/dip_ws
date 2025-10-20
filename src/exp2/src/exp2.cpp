#include <stdlib.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include <geometry_msgs/Twist.h>
#include "sensor_msgs/Image.h"
#include <math.h>
#include <cv_bridge/cv_bridge.h>

enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = COMPUTER;
#define pi 3.1415926
using namespace cv;

//空域均值滤波函数
void meanFilter(Mat &input)
{

    //生成模板
    int T_size = 9; 
    //int T_size = 3;                                   // 模板大小
    Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
    /*** 第一步：在此处填充均值滤波模板 ***/

    double mean_value = 1.0 / (T_size * T_size);
    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {
            Template.at<double>(i, j) = mean_value;
        }
    }


    // 卷积
    Mat output = Mat::zeros(input.size(), CV_64F);

    /*** 第二步：填充模板与输入图像的卷积代码 ***/    
    int half_T = T_size / 2;
    for (int i = half_T; i < input.rows; i++)
    {
        for (int j = half_T; j < input.cols; j++)
        {
            for (int m = 0; m < T_size; m++)
            {
                for (int n = 0; n < T_size; n++)
                {
                    output.at<double>(i, j) += Template.at<double>(m, n) * input.at<uchar>(i + m - half_T, j + n - half_T);
                }
            }
        }
    }


    output.convertTo(output, CV_8UC1);
    imshow("mean_filtered_image", output);
}
// 空域高斯滤波器函数
void gaussianFilter(Mat &input, double sigma)
{

    //利用高斯函数生成模板
    int T_size = 9;                                    // 模板大小
    Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
    int center = round(T_size / 2);                    // 模板中心位置
    double sum = 0.0;
    
    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {

            /*** 第三步：在此处填充高斯滤波模板元素计算代码 ***/
            Template.at<double>(i, j) = exp(-(pow(i - center, 2) + pow(j - center, 2)) / (2 * pow(sigma, 2))) / (2 * pi * pow(sigma, 2));
            sum += Template.at<double>(i, j); //用于归一化模板元素
        }
    }


    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {

            /*** 第四步：在此处填充模板归一化代码 ***/
            Template.at<double>(i, j) /= sum;
        }
    }
    // 卷积
    Mat output = Mat::zeros(input.size(), CV_64F);

    /*** 第五步：同第二步，填充模板与输入图像的卷积代码 ***/ 
    int half_T = T_size / 2;
    for (int i = half_T; i < input.rows; i++)
    {
        for (int j = half_T; j < input.cols; j++)
        {
            for (int m = 0; m < T_size; m++)
            {
                for (int n = 0; n < T_size; n++)
                {
                    output.at<double>(i, j) += Template.at<double>(m, n) * input.at<uchar>(i + m - half_T, j + n - half_T);
                }
            }
        }
    }


    output.convertTo(output, CV_8UC1);
    imshow("spatial_filtered_image", output);
}
// 锐化空域滤波
void sharpenFilter(Mat &input)
{

    //生成模板
    int T_size = 3;                                    // 模板大小
    Mat Template = Mat::zeros(T_size, T_size, CV_64F); // 初始化模板矩阵
    /*** 第六步：填充锐化滤波模板 ***/   
    Template.at<double>(0, 0) = 0;   Template.at<double>(0, 1) = -1;  Template.at<double>(0, 2) = 0;
    Template.at<double>(1, 0) = -1;  Template.at<double>(1, 1) = 5;   Template.at<double>(1, 2) = -1;
    Template.at<double>(2, 0) = 0;   Template.at<double>(2, 1) = -1;  Template.at<double>(2, 2) = 0;

    // 卷积
    Mat output = Mat::zeros(input.size(), CV_64F);

    /*** 第七步：同第二步，填充模板与输入图像的卷积代码 ***/    
    int half_T = T_size / 2;
    for (int i = half_T; i < input.rows; i++)
    {
        for (int j = half_T; j < input.cols; j++)
        {
            for (int m = 0; m < T_size; m++)
            {
                for (int n = 0; n < T_size; n++)
                {
                    output.at<double>(i, j) += Template.at<double>(m, n) * input.at<uchar>(i + m - half_T, j + n - half_T);
                }
            }
        }
    }



    output.convertTo(output, CV_8UC1);
    imshow("sharpen_filtered_image", output);
}
// 膨胀函数
void Dilate(Mat &Src)
{
    Mat Dst = Src.clone();
    Dst.convertTo(Dst, CV_64F);

    /*** 第八步：填充膨胀代码 ***/    
    int T_size = 9; // 使用9x9的结构元素
    int half_T = T_size / 2;
    
    Mat Template = Mat::zeros(T_size, T_size, CV_8UC1); 
    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {
            Template.at<uchar>(i, j) = 1;
        }
    }
    //膨胀黑色部分(有一个黑则变黑，否则不变)
    for(int i = half_T; i < Src.rows - half_T; i++) 
    {
        for(int j = half_T; j < Src.cols - half_T; j++) 
        {
            bool hit = false;
            for (int m = 0; m < T_size && (!hit); m++)
            {
                for (int n = 0; n < T_size && (!hit); n++)
                {
                    if (Src.at<uchar>(i+m, j+n) == 0 && Template.at<uchar>(m,n))
                    {
                        hit = true;
                        break;
                    }
                }
            }
            if (hit)
                Dst.at<double>(i, j) = 0;
            else
                Dst.at<double>(i, j) = 255;
        }
    }
    

    Dst.convertTo(Dst, CV_8UC1);
    imshow("dilate", Dst);

        // 遍历图像每个像素
    // for(int i = half_T; i < Src.rows - half_T; i++) 
    // {
    //     for(int j = half_T; j < Src.cols - half_T; j++) 
    //     {
    //         double maxVal = 0;
    //         // 在3x3邻域内找最大值
    //         for(int m = -half_T; m <= half_T; m++) 
    //         {
    //             for(int n = -half_T; n <= half_T; n++) 
    //             {
    //                 maxVal = max(maxVal, (double)Src.at<uchar>(i+m, j+n));
    //             }
    //         }
    //         Dst.at<double>(i,j) = maxVal;
    //     }
    // }
}
// 腐蚀函数
void Erode(Mat &Src)
{
    Mat Dst = Src.clone();
    Dst.convertTo(Dst, CV_64F);


    /*** 第九步：填充腐蚀代码 ***/    
    int T_size = 9; // 使用9x9的结构元素
    int half_T = T_size / 2;
    Mat Template = Mat::zeros(T_size, T_size, CV_8UC1); 
    for (int i = 0; i < T_size; i++)
    {
        for (int j = 0; j < T_size; j++)
        {
            Template.at<uchar>(i, j) = 1;
        }
    }
    //腐蚀黑色部分(全为黑则不改变，否则变白)
    for(int i = half_T; i < Src.rows - half_T; i++) 
    {
        for(int j = half_T; j < Src.cols - half_T; j++) 
        {
            bool hit = true;
            for (int m = 0; m < T_size && hit; m++)
            {
                for (int n = 0; n < T_size && hit; n++)
                {
                    if (Src.at<uchar>(i+m, j+n) == 255 && Template.at<uchar>(m,n))
                    {
                        hit = false;
                        break;
                    }
                }
            }
            if (hit)
                Dst.at<double>(i, j) = 0;
            else
                Dst.at<double>(i, j) = 255;
        }
    }
    



    Dst.convertTo(Dst, CV_8UC1);
    imshow("erode", Dst);

        // 遍历图像每个像素
    // for(int i = half_T; i < Src.rows - half_T; i++) {
    //     for(int j = half_T; j < Src.cols - half_T; j++) {
    //         double minVal = 255;
    //         // 在3x3邻域内找最小值
    //         for(int m = -half_T; m <= half_T; m++) {
    //             for(int n = -half_T; n <= half_T; n++) {
    //                 minVal = min(minVal, (double)Src.at<uchar>(i+m, j+n));
    //             }
    //         }
    //         Dst.at<double>(i,j) = minVal;
    //     }
    // }
}

Mat frame_msg;
void rcvCameraCallBack(const sensor_msgs::Image::ConstPtr& img)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    frame_msg = cv_ptr->image;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "exp2_node"); // 初始化 ROS 节点
    ros::NodeHandle n;
    ros::Subscriber camera_sub;
    VideoCapture capture;
    capture.open(0);     

    if(state == COMPUTER)
    {
        capture.open(0);     
        if (!capture.isOpened())
        {
            printf("电脑摄像头没有正常打开\n");
            return 0;
        }
        waitKey(1000);
    }
    else if(state == ZED)
    {
        capture.open(4);     
        if (!capture.isOpened())
        {
            printf("ZED摄像头没有正常打开\n");
            return 0;
        }
        waitKey(1000);
    }
    else if(state == REALSENSE)
    {
        camera_sub = n.subscribe("/camera/color/image_raw",1,rcvCameraCallBack);
    }

    Mat frIn;                                        // 当前帧图片
    Mat test1;
    Mat test2;
    while (ros::ok())
    {

        if(state == COMPUTER)
        {
            capture.read(frIn);
            if (frIn.empty())
            {
                printf("没有获取到电脑图像\n");
                continue;
            }
        }
        else if(state == ZED)
        {
            capture.read(frIn);
            if (frIn.empty())
            {
                printf("没有获取到ZED图像\n");
                continue;
            }
            frIn = frIn(cv::Rect(0,0,frIn.cols/2,frIn.rows));//截取zed的左目图片
        }
        else if(state == REALSENSE)
        {
            if(frame_msg.cols == 0)
            {
                printf("没有获取到realsense图像\n");
                ros::spinOnce();
                continue;
            }
            frIn = frame_msg;
        }

	test1 = imread("//home//eaibot//test1.png", IMREAD_COLOR);
	test2 = imread("//home//eaibot//test2.png", IMREAD_COLOR);
	
	cvtColor(test1, test1, COLOR_BGR2GRAY);
	cvtColor(test2, test2, COLOR_BGR2GRAY);
	

	
    for(int i = 0; i < test2.rows; i++) 
    {
	for(int j = 0; j < test2.cols; j++) 
	{
		if(test2.at<uchar>(i, j)>127)
		{
			test2.at<uchar>(i, j) = 255;
		}
		else
		{
			test2.at<uchar>(i, j) = 0;
		}
	}
    }
	imshow("original_image_1", test1);
	imshow("original_image_2", test2);
	
        //cvtColor(frIn, frIn, COLOR_BGR2GRAY);
        //imshow("original_image_1", frIn);
        //空域均值滤波
	meanFilter(test1);
	
        // 空域高斯滤波
        double sigma = 2.5;
        gaussianFilter(test1, sigma);

        //空域锐化滤波
        sharpenFilter(test1);

        // 膨胀函数
        Dilate(test2);

        // 腐蚀函数
        Erode(test2);

        ros::spinOnce();
        waitKey(5);
    }
    return 0;
}
