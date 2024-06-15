#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

double calculateAngle(cv::Mat &image , cv::Point &point1)
{
    //initialise all empty variables and matrices
    cv::Mat img;
    cv::Mat th1;
    cv::Mat th2;
    cv::Mat unknown;
    std::vector<std::vector<cv::Point>> contours;

    //convert input image to grayscale, binary, and then perform opening morphology to eliminate noise signals
    cv::cvtColor(image,img,cv::COLOR_BGR2GRAY);
    cv::threshold(img,th1,160,255,cv::THRESH_BINARY);
    int morph_size = 3; 
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size)); 
    cv::morphologyEx(th1, th2, cv::MORPH_OPEN, element, cv::Point(-1, -1), 2); 

    //find single connected region in binary image and using contours and mark the largest contour object with green color
    cv::findContours(th2, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> largestContour = *max_element(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) { return contourArea(a) < contourArea(b);});
    drawContours(image, std::vector<std::vector<cv::Point>>{largestContour}, -1, cv::Scalar(0,255,0), 3);

    //get the datapoints of the largest contour, and calculate the principle component analyses to get the rotational angle of the object
    cv::Mat datapts= cv::Mat(largestContour).reshape(1,largestContour.size());
    datapts.convertTo(datapts, CV_32F);
    cv::PCA pca_analysis(datapts,unknown, cv::PCA::DATA_AS_ROW);
    cv::Point2f mean = cv::Point2f(pca_analysis.mean.at<float>(0, 0), pca_analysis.mean.at<float>(0, 1));
    cv::Point2f eigen_vec = cv::Point2f(pca_analysis.eigenvectors.at<float>(0, 0), pca_analysis.eigenvectors.at<float>(0, 1));
    point1= cv::Point(mean.x,mean.y);
    cv::Point p1 = cv::Point(cvRound(mean.x + 0.02f * eigen_vec.x * 1000), cvRound(mean.y + 0.02f * eigen_vec.y * 1000));
    cv::Point p2 = cv::Point(cvRound(mean.x - 0.02f * eigen_vec.x * 1000), cvRound(mean.y - 0.02f * eigen_vec.y * 1000));
    
    //calculate the angle of rotation in degrees
    double angle = atan2(eigen_vec.y, eigen_vec.x);
    double angle_degrees = angle * 180.0 / CV_PI;
    //return the angle
    return(angle_degrees);
}

double watershedAnalysis(cv::Mat image, cv::Point &point1)
{
    //initialise all empty variables and matrices
    cv::Mat img;
    cv::Mat th1;
    cv::Mat dist;
    cv::Mat th2;
    cv::Mat unknown;
    cv::Mat unknown2;
    cv::Mat markers;
    cv::Mat markers_8u;
    std::vector<std::vector<cv::Point>> contours;

    //convert input image to grayscale, binary, and then perform opening morphology to eliminate noise signals
    cv::cvtColor(image,img,cv::COLOR_BGR2GRAY);
    cv::threshold(img,th1,160,255,cv::THRESH_BINARY);
    int morph_size = 3; 
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size)); 
    cv::morphologyEx(th1, th1, cv::MORPH_OPEN, element, cv::Point(-1, -1), 2); 
    cv::dilate(th1, dist, element,cv::Point(-1,-1), 3);

    //calculated the distance transform, normalised, thresholded and converted the image to 8 bit unsigned format
    cv::distanceTransform(dist,dist,cv::DIST_L2, 5);
    cv::normalize(dist,dist, 0, 1.0, cv::NORM_MINMAX);
    cv::threshold(dist, th2, 0.2, 255,0);
    th2.convertTo(th2,CV_8U);

    //subtracted th1 from th2 to elimate regions that does not either form the for ground or background
    cv::subtract(th1, th2, unknown);
    markers = cv::Mat::zeros(th1.size(), CV_32S);
    connectedComponents(th2, markers);
    markers += 1;
    markers.setTo(0, unknown == 255);

    //calculate the watershed algorithm and convert the markers to 8 bit unsigned integer
    cv::watershed(image,markers);
    image.setTo(cv::Scalar(0,0,255), markers==-1); 
    markers.convertTo(markers_8u, CV_8U);

    //find single connected region in binary image and using contours and mark the largest contour object with green color
    findContours(markers_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> largestContour = *max_element(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) { return contourArea(a) < contourArea(b);});
    drawContours(image, std::vector<std::vector<cv::Point>>{largestContour}, -1, cv::Scalar(0,255,0), 3);

    //get the datapoints of the largest contour, and calculate the principle component analyses to get the rotational angle of the object
    cv::Mat datapts= cv::Mat(largestContour).reshape(1,largestContour.size());
    datapts.convertTo(datapts, CV_32F);
    cv::PCA pca_analysis(datapts,unknown2, cv::PCA::DATA_AS_ROW);
    cv::Point2f mean = cv::Point2f(pca_analysis.mean.at<float>(0, 0), pca_analysis.mean.at<float>(0, 1));
    cv::Point2f eigen_vec = cv::Point2f(pca_analysis.eigenvectors.at<float>(0, 0), pca_analysis.eigenvectors.at<float>(0, 1));
    point1= cv::Point(mean.x,mean.y);
    cv::Point p1 = cv::Point(cvRound(mean.x + 0.02f * eigen_vec.x * 1000), cvRound(mean.y + 0.02f * eigen_vec.y * 1000));
    cv::Point p2 = cv::Point(cvRound(mean.x - 0.02f * eigen_vec.x * 1000), cvRound(mean.y - 0.02f * eigen_vec.y * 1000));

    // //calculate the angle of rotation in degrees
    // double angle = atan2(eigen_vec.y, eigen_vec.x);
    // double angle_degrees = angle * 180.0 / CV_PI;

    // //return the angle
    // return(angle_degrees);
    return(0.9);
}

int main(int argc, char *argv[])
{   
    //read the test image and template image from command line arguments, initialise empty points to store mean value of objects in the image
    cv::Mat test_img = cv::imread(argv[1]);
    cv::Mat temp_img = cv::imread(argv[2]);
    cv::Point mean1;
    cv::Point mean2;

    //calculate angle of object in both the image
    // double test_angle = calculateAngle(test_img, mean1);
    // double temp_angle = calculateAngle(temp_img, mean2);
    double test_angle = watershedAnalysis(test_img, mean1);
    double temp_angle = watershedAnalysis(temp_img, mean2);

    //calculate the rotational angle between the object in test image and object in template image
    double Angle = test_angle - temp_angle;
    printf("%f-%f=%f\n",test_angle,temp_angle,Angle);
    if (Angle < 0) Angle += 360.0;

    //display the test image with the text containing the rotational angle 
    // cv::putText(temp_img,std::to_string(Angle), cv::Point(mean2.x, mean2.y),cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
    cv::imshow("Template Image", temp_img);
    cv::waitKey(0);
}