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

#define PI 3.1415926

enum CameraState
{
    COMPUTER = 0,
    ZED,
    REALSENSE
};
CameraState state = COMPUTER;


using namespace cv;
using namespace std;
void Gaussian(const Mat &input, Mat &output, double sigma)
{
    if (output.rows != input.rows || output.cols != input.cols || output.channels() != input.channels())
        return;
    int kernel_size = 9;
    int center = kernel_size / 2;
    double gaussian_kernel[kernel_size][kernel_size];
    double sum = 0;

    /*** 第一步：结合实验二，在此处填充高斯滤波代码 ***/
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            gaussian_kernel[i][j] = exp(-(pow(i - center, 2) + pow(j - center, 2)) / (2 * pow(sigma, 2))) / (2 * PI * pow(sigma, 2));
            sum += gaussian_kernel[i][j];
        }
    }
    
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            gaussian_kernel[i][j] /= sum;
        }
    }

    for(int i = center; i < input.rows - center; i++)
    {
        for (int j = center; j < input.cols - center; j++)
        {
            Vec3d sum(0, 0, 0);
            for (int m = -center; m <= center; m++)
            {
                for (int n = -center; n <= center; n++)
                {
                    Vec3b pixel = input.at<Vec3b>(i+m, j+n);
                    sum[0] += pixel[0] * gaussian_kernel[m+center][n+center];
                    sum[1] += pixel[1] * gaussian_kernel[m+center][n+center];
                    sum[2] += pixel[2] * gaussian_kernel[m+center][n+center];
                }
            }
            output.at<Vec3b>(i, j)[0] = (int)sum[0];
            output.at<Vec3b>(i, j)[1] = (int)sum[1];
            output.at<Vec3b>(i, j)[2] = (int)sum[2];
        }
    }



}

// void BGR2HSV(const Mat &input, Mat &output)
// {
//     if (input.rows != output.rows ||
//         input.cols != output.cols ||
//         input.channels() != 3 ||
//         output.channels() != 3)
//         return;


// 	for(int i = 0; i < input.rows; i++)
// 	{
// 		for (int j = 0; j < input.cols; j++)
// 		{

//             /*** 第二步：在此处填充RGB转HSV代码 ***/
//             Vec3b pixel = input.at<Vec3b>(i, j);   
//             double max_bgr = max(pixel[0], max(pixel[1], pixel[2]));
//             double min_bgr = min(pixel[0], min(pixel[1], pixel[2]));
//             double delta = max_val - min_val;
//             //HSV
            
//             double h,s,v;
//             v= max_bgr;
//             if (v){
//                 s = delta / max_val;
//             }
//             else{
//                 s = 0;
//             }

//             if (delta != 0) {
//                 if (max_val == r) 
//                 {
//                     h = 60 * fmod((g - b) / delta, 6);
//                 } else if (max_val == g) 
//                 {
//                     h = 60 * ((b - r) / delta + 2);
//                 } else 
//                 {
//                     h = 60 * ((r - g) / delta + 4);
//                 }

//             if (h < 0)200
//             {
//                 h += 360;
//             }
//             output.at<Vec3b>(i, j)[0] = (int)(h/2.0);
//             output.at<Vec3b>(i, j)[1] = (int)(s*255.0);
//             output.at<Vec3b>(i, j)[2] = (int)(v*255.0);
//             }
//         }
//     }
// }

void BGR2HSI(const Mat &input, Mat &output)
{
    if (input.rows != output.rows ||
        input.cols != output.cols ||
        input.channels() != 3 ||
        output.channels() != 3)
        return;


	for(int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{

            /*** 第二步：在此处填充RGB转HSI代码 ***/
            Vec3b pixel = input.at<Vec3b>(i, j);   
            double max_bgr = max(pixel[0], max(pixel[1], pixel[2]));
            double min_bgr = min(pixel[0], min(pixel[1], pixel[2]));
            //HSI
            
            double h,s,I;
            I = (pixel[0] + pixel[1] + pixel[2])/3.0;
            s = 1 - 3*min_bgr/(pixel[0] + pixel[1] + pixel[2]);
            h = acos((pixel[2]-pixel[1] + pixel[2]-pixel[0])/(2*sqrt(pow(pixel[2]-pixel[1],2) + (pixel[2]-pixel[0])*(pixel[1]-pixel[0])) + 0.00001));
            if (pixel[0] > pixel[1])
            {
                h = 2*PI - h;
            }

            output.at<Vec3b>(i, j)[0] = (int)(h*255.0/PI/2.0);
            output.at<Vec3b>(i, j)[1] = (int)(s*255.0);
            output.at<Vec3b>(i, j)[2] = (int)I;
        }
    }
}


void ColorSplitManual(const Mat &hsv_input, Mat &grey_output, const string window)
{
	static int hmin = 0;
	static int hmax = 255;
	static int smin = 0;
	static int smax = 255;
	static int imin = 0;
	static int imax = 255;
	createTrackbar("Hmin", window, &hmin, 255);
	createTrackbar("Hmax", window, &hmax, 255);
	createTrackbar("Smin", window, &smin, 255);
	createTrackbar("Smax", window, &smax, 255);
	createTrackbar("Vmin", window, &imin, 255);
	createTrackbar("Vmax", window, &imax, 255);

    /*** 第三步：在此处填充阈值分割代码代码 ***/
	int row = hsv_input.rows;
	int col = hsv_input.cols;
	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			if(imin <= hsv_input.at<Vec3b>(i, j)[2] && hsv_input.at<Vec3b>(i, j)[2] <= imax &&
			   smin <= hsv_input.at<Vec3b>(i, j)[1] && hsv_input.at<Vec3b>(i, j)[1] <= smax &&
			   hmin <= hsv_input.at<Vec3b>(i, j)[0] && hsv_input.at<Vec3b>(i, j)[0] <= hmax)
				grey_output.at<uchar>(i, j) = 255;
			else grey_output.at<uchar>(i, j) = 0;
		}
	}
}

void ColorSplitAuto(const Mat &hsv_input, Mat &bgr_output, vector<vector<Point>> &contours, int hmin, int hmax, int smin, int smax, int vmin, int vmax)
{
    int rw = hsv_input.rows;
	int cl = hsv_input.cols;
    Mat color_region(rw, cl, CV_8UC1);
    int flag = 0;
    if (hmax == 255)
        flag = 1;
    /*** 第五步：利用已知的阈值获取颜色区域二值图 ***/
    for (int i = 0; i < rw; i++)
    {
        for (int j = 0; j < cl; j++)
        {
            if(vmin <= hsv_input.at<Vec3b>(i, j)[2] && hsv_input.at<Vec3b>(i, j)[2] <= vmax &&
               smin <= hsv_input.at<Vec3b>(i, j)[1] && hsv_input.at<Vec3b>(i, j)[1] <= smax &&
               hmin <= hsv_input.at<Vec3b>(i, j)[0] && hsv_input.at<Vec3b>(i, j)[0] <= hmax
            )
                color_region.at<uchar>(i, j) = 255;
            else{
                color_region.at<uchar>(i, j) = 0;
            }
            if (flag)
            {
                if(0 <= hsv_input.at<Vec3b>(i, j)[2] && hsv_input.at<Vec3b>(i, j)[2] <= 25 &&
               smin <= hsv_input.at<Vec3b>(i, j)[1] && hsv_input.at<Vec3b>(i, j)[1] <= smax &&
               hmin <= hsv_input.at<Vec3b>(i, j)[0] && hsv_input.at<Vec3b>(i, j)[0] <= hmax
            )
                color_region.at<uchar>(i, j) = 255;
            }
            
        }
    }
    // imshow("color_region", color_region);



    /* 获取多边形轮廓 */
    vector<Vec4i> hierarchy;
	findContours(color_region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	vector<vector<Point>> lines(contours.size());
    /* 利用多项式近似平滑轮廓 */
	for(int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], lines[i],9,true);
	}
	drawContours(bgr_output, lines, -1,Scalar(0, 0, 255), 2, 8);

}


void GetROI(const Mat &input, Mat &output, const vector<vector<Point>> &contour)
{
    /* 第六步：补充获取颜色区域代码，可使用drawContours函数 */
    Mat mask = Mat::zeros(input.size(), CV_8UC1);

    drawContours(mask, contour, -1, Scalar(255, 255, 255), FILLED, 8);
    input.copyTo(output, mask);
    // imshow("ROI", output);
}

int CountROIPixel(const Mat &input)
{
	int cnt = 0;

    /* 第七步：补充获取颜色区域像素个数的代码 */
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            Vec3b pixel = input.at<Vec3b>(i, j);
            if (pixel[0] > 0 && pixel[1] > 0 && pixel[2] > 0)
            {
                cnt++;
            }
        
        }
    }
    return cnt;
}

/*** 第四步：在第三步基础上修改各颜色阈值 ***/
//{hmin, hmax, smin, smax, vmin, vmax}
int red_thresh[6] = {200,255,44,255,46,255};
int green_thresh[6] = {60,100,44,255,46,255};
int blue_thresh[6] = {120,180,44,255,46,255};
int yellow_thresh[6] = {30,60,44,255,46,255};

Mat frame_msg;
void rcvCameraCallBack(const sensor_msgs::Image::ConstPtr& img)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    frame_msg = cv_ptr->image;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "exp4_node"); // 初始化 ROS 节点
    ros::NodeHandle n;
	ros::Publisher vel_pub;
    ros::Subscriber camera_sub;
    VideoCapture capture;
    vel_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
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
        }n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
        waitKey(1000);
    }
    else if(state == REALSENSE)
    {
        camera_sub = n.subscribe("/camera/color/image_raw",1,rcvCameraCallBack);
    }
    

    Mat frIn;
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


        // 空域高斯滤波
        Mat filter(frIn.size(), CV_8UC3);
        Gaussian(frIn, filter, 3);
        imshow("filter",filter);

        // RGB转HSI
        Mat hsi(frIn.size(), CV_8UC3);
        BGR2HSI(filter, hsi);
        imshow("hsi",hsi);

        // 手动颜色分割
        Mat grey(frIn.rows, frIn.cols, CV_8UC1);
        ColorSplitManual(hsi, grey, "hsi");
        imshow("split", grey);
        
        int colors = 0;
        int maxs_color_num = 0;
        /* 目标颜色检测 */

	    Mat tmp_line = frIn.clone();
	    Mat tmp_roi = Mat::zeros(frIn.size(), CV_8UC3);
        vector<vector<Point>> contours_r;
        	ColorSplitAuto(hsi, tmp_line, contours_r, red_thresh[0], red_thresh[1], red_thresh[2],
				   red_thresh[3], red_thresh[4], red_thresh[5]);
	    GetROI(frIn, tmp_roi, contours_r);
	    int red_color_num = CountROIPixel(tmp_roi);

        /* 第八步：结合给出的检测红颜色的代码框架，给出控制小车运动的代码 */

        // 检测绿色
        Mat tmp_line_g = frIn.clone();
        Mat tmp_roi_g = Mat::zeros(frIn.size(), CV_8UC3);
        vector<vector<Point>> contours_g;
        ColorSplitAuto(hsi, tmp_line_g, contours_g, green_thresh[0], green_thresh[1], green_thresh[2],
                       green_thresh[3], green_thresh[4], green_thresh[5]);
        GetROI(frIn, tmp_roi_g, contours_g);
        imshow("line", tmp_line_g);
        int green_color_num = CountROIPixel(tmp_roi_g);
        
        // 检测蓝色
        Mat tmp_line_b = frIn.clone();
        Mat tmp_roi_b = Mat::zeros(frIn.size(), CV_8UC3);
        vector<vector<Point>> contours_b;
        ColorSplitAuto(hsi, tmp_line_b, contours_b, blue_thresh[0], blue_thresh[1], blue_thresh[2],
                       blue_thresh[3], blue_thresh[4], blue_thresh[5]);
        GetROI(frIn, tmp_roi_b, contours_b);
        int blue_color_num = CountROIPixel(tmp_roi_b);
        
        // 检测黄色
        Mat tmp_line_y = frIn.clone();
        Mat tmp_roi_y = Mat::zeros(frIn.size(), CV_8UC3);
        vector<vector<Point>> contours_y;
        ColorSplitAuto(hsi, tmp_line_y, contours_y, yellow_thresh[0], yellow_thresh[1], yellow_thresh[2],
                       yellow_thresh[3], yellow_thresh[4], yellow_thresh[5]);
        GetROI(frIn, tmp_roi_y, contours_y);
        int yellow_color_num = CountROIPixel(tmp_roi_y);
        

        maxs_color_num = red_color_num;
        colors = 0;
        if (green_color_num > maxs_color_num)
        {
            colors = 1;
            maxs_color_num = green_color_num;
        }
        if (blue_color_num > maxs_color_num)
        {
            colors = 2;
            maxs_color_num = blue_color_num;
        }
        if (yellow_color_num > maxs_color_num)
        {
            colors = 3;
            maxs_color_num = yellow_color_num;
        }

        if (maxs_color_num < 60000)
        { 
            maxs_color_num = 0;
        }
        ROS_INFO("red_color_num: %d", red_color_num);
        ROS_INFO("green_color_num: %d", green_color_num);
        ROS_INFO("blue_color_num: %d", blue_color_num);
        ROS_INFO("yellow_color_num: %d", yellow_color_num);
        ROS_INFO("maxs_color_num: %d", maxs_color_num);
        ROS_INFO("colors: %d", colors);


        geometry_msgs::Twist vel;
        vel.linear.x = 0;
        vel.linear.y = 0;
        vel.linear.z = 0;
        vel.angular.x = 0;
        vel.angular.y = 0;
        vel.angular.z = 0;
        if(maxs_color_num)
        {
            switch(colors)
            {
                case 0:
                    vel.linear.x = 0.4;
                    break;
                case 1:
                    vel.linear.x = -0.4;
                    break;
                case 2:
                    vel.angular.z = 0.4;
                    break;
                case 3:
                    vel.angular.z = -0.4;
                    break;
            }
        }
        else {
            vel.linear.x = 0;
            vel.angular.z = 0;
        }
        vel_pub.publish(vel);
        ros::spinOnce();
        waitKey(5);
    }
    return 0;
}