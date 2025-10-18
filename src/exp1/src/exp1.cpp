#include <stdlib.h>

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ros/ros.h"
#include "iostream"
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/Twist.h"


enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = REALSENSE;

ros::Publisher vel_pub;

#define N 255 //灰度level
using namespace std;
using namespace cv;



//getHistImage()--画图像直方图
Mat getHistImage( Mat hist)
{
    Scalar color(172, 172, 100);//划线颜色
    Scalar Background(255,255,255);//背景颜色
    int thickness = 2;	//划线宽度
    int histss[256] = {0};

    Mat dstHist;
    /*** 第一步：下面计算不同灰度值的像素分布 ***/

    // calcHist(&hist, 1, 0, Mat(), dstHist, 1, {256}, {0, 256});//计算直方图
    //提取到数组中
    // for (int i = 0; i < 256; i++) {
    //     histss[i] = (int)dstHist.at<float>(i,0);
    // }
    for(int i = 0; i < hist.rows; i++) {
        for(int j = 0; j < hist.cols; j++) {
            int grayValue = (int)hist.at<uchar>(i, j);
            histss[grayValue]++;
        }
    }
    int maxValue = 0;
    for (int i = 0; i < 256; i++) {
        if (histss[i] > maxValue) {
            maxValue = histss[i];
        }
    }

    int histSize = 270;
    Mat histImage(histSize, histSize, CV_8UC3, Background );//绘制背景

    for (int h = 0; h < 256; h++) {

    /*** 第二步：画出像素的直方图分布 ***/
    int height = (int)((float)histss[h] / maxValue * histSize);
    line(histImage, Point(h, histSize), Point(h, histSize - height), color, thickness);


    }
    return histImage;
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
	ros::init(argc, argv, "exp1_node"); // 初始化 ROS 节点
    ros::NodeHandle n;
    ros::Subscriber camera_sub;
    
    vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);// 发布速度消息
	VideoCapture capture;

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
		waitKey(1000);
	}

	Mat frame;//当前帧图片

	int Grayscale[N];//灰度级
	int Grayscale2[N];//均衡化以后的灰度级
	float Gray_f[N];//频率
	int Gray_c[N];//累计密度
    ros::Rate loop_rate(10); // 设置循环频率为10Hz
	while (ros::ok())
	{
        if(state == COMPUTER)
        {
            capture.read(frame);
            if (frame.empty())
            {
                printf("没有获取到电脑图像\n");
                continue;
            }
        }
        else if(state == ZED)
        {
            capture.read(frame);
            if (frame.empty())
            {
                printf("没有获取到ZED图像\n");
                continue;
            }
            frame = frame(cv::Rect(0,0,frame.cols/2,frame.rows));//截取zed的左目图片
        }
        else if(state == REALSENSE)
        {
            if(frame_msg.cols == 0)
            {
                printf("没有获取到realsense图像\n");
                ros::spinOnce();
                continue;
            }
			frame = frame_msg;
        }


		Mat frIn = frame; 
		Mat New;
		cvtColor(frIn,frIn,COLOR_RGB2GRAY,0);

    	/*** 第三步：直方图均衡化处理 ***/
        // New = equalizeHist(frIn, New);
	int histdata[256] = {0};
	double Cr[256] = {0.0};
	int row = frIn.rows;
	int col = frIn.cols;
	for(int i = 0; i < row; i++) {
        	for(int j = 0; j < col; j++) {
            		int grayValue = (int)frIn.at<uchar>(i, j);
            		histdata[grayValue]++;
        	}
    	}
    	
    	for(int i = 0; i < 256; i++) {
    		if (i){
    			Cr[i] = Cr[i-1] + (double)histdata[i]/row/col;
    		}
    		else{
    			Cr[i] = (double)histdata[i]/row/col;
    		}
    		
    	}
    	int S_min;
    	int S_max;
    	
    	for(int i = 0; i < 256; i++) {
    		if (histdata[i]){
    			S_min = i;
    			break;
    		}
    	}
    	for(int i = 255; i >= 0; i--) {
    		if (histdata[i]){
    			S_max = i;
    			break;
    		}
    	}
    	
    	New = Mat::zeros(row,col,CV_8UC1);
 	for(int i = 0; i < row; i++) {
        	for(int j = 0; j < col; j++) {
            		New.at<uchar>(i,j) = (int)(Cr[(int)frIn.at<uchar>(i, j)] * (S_max-S_min) + S_min);
        	}
    	}   	
    	
	
	Mat last = getHistImage(New);
	Mat origi= getHistImage(frIn);
	imshow("his",last);//均衡化后直方图
	imshow("origi",origi);//原直方图
	imshow("Histed",New);//均衡化后图像
	imshow("Origin",frIn);//原图像


    	/*** 第四步：参考demo程序，添加让小车原地旋转代码 ***/
        geometry_msgs::Twist vel_msg;
        // 发布消息，让小车绕一小半径旋转
        vel_msg.linear.x = 0.05;
        vel_msg.linear.y = 0.;
        vel_msg.angular.z = 0.5;
        vel_pub.publish(vel_msg);



        ros::spinOnce(); // 处理回调函数
        waitKey(5);
        loop_rate.sleep(); // 控制循环速率
	
	}
	return 0;
}
