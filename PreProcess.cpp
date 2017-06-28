#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <bits/stdc++.h>




using namespace cv;
using namespace std;

// Classifies image and adds up total value
int FindValue(Mat& img, float W[][4], float b[])
{
    
    int i,j,val=0;
    float score[4];
    Mat flat = img.reshape(1,1);
    

    for(i=0;i<4;i++)
    {
        val=0;
        for(j=0;j<900;j++)
            val+=W[j][i]*(int)flat.at<uchar>(0,j);
        score[i] = val + b[i];
    }

    cout<<"Scores :"<<endl;
    for(i=0;i<4;i++)
        cout<<score[i]<<endl;
    int mxm=score[0],max=0;
    for(i=0;i<4;i++)
        if(score[i]>mxm) max=i;

    if(max == 0 ) return 1;

    if(max == 1 ) return 2;

    if(max == 2 ) return 5;

    if(max == 3 ) return 10;

}




int main(int argc, char** argv)
{
    //file in which result is written
	ofstream myfile;
    myfile.open ("example.txt");
    
	if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }
	

	Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

      // Create a new matrix to hold the HSV image
    


    Mat HSV,Gray,Gray2;
    Mat img,dst;
    int morph_elem = 0;
    int morph_size = 0;
    int morph_operator = 0;
    int const max_operator = 4;
    int const max_elem = 2;
    int const max_kernel_size = 21;


    int threshold_value = 0;
    int threshold_type = 3;
    int const max_value = 255;
    int const max_type = 4;
    int const max_BINARY_value = 255;
    
    // convert RGB image to HSV
    cvtColor(image, HSV, CV_BGR2HSV);
    cvtColor(image, Gray, CV_BGR2GRAY);
    cvtColor(image, Gray2, CV_BGR2GRAY);


    //namedWindow("Display window", CV_WINDOW_AUTOSIZE);
    //imshow("Display window", image);

    //namedWindow("Result window", CV_WINDOW_AUTOSIZE);
    //imshow("Result window", HSV);


    vector<Mat> hsv_planes;
    split(HSV, hsv_planes);
    Mat h = hsv_planes[0]; // H channel
    Mat s = hsv_planes[1]; // S channel
    Mat v = hsv_planes[2]; // V channel

    //namedWindow("hue", CV_WINDOW_AUTOSIZE);
    //imshow("hue", h);
    //namedWindow("saturation", CV_WINDOW_AUTOSIZE);
    //imshow("saturation", s);
    //namedWindow("value", CV_WINDOW_AUTOSIZE);
    //imshow("value", v);
    int peaksat=0,peakhue=0;
    
    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            if(s.at<uchar>(i,j) > peaksat)
                peaksat = s.at<uchar>(i,j);

            if(h.at<uchar>(i,j) > peakhue)
                peakhue = h.at<uchar>(i,j);
          
            //cout<<(int)intensity.val[0]<<" ";
        }
        
        //cout<<endl;
    }

    Mat threshrating(h);

    float thresh=0;
    float k=0;

    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            
            threshrating.at<uchar>(i,j) = s.at<uchar>(i,j) - peaksat + abs(h.at<uchar>(i,j) - peakhue);
            //cout<<(float)threshrating.at<uchar>(i,j)<<" ";
            thresh = (thresh*k + (float)threshrating.at<uchar>(i,j) )/(k+1);
            k++;
            //cout<<(int)intensity.val[0]<<" ";
        }
        
        //cout<<endl;
    }

    for(int i=0;i<image.rows;i++)
    {
        for(int j=0;j<image.cols;j++)
        {
            
            if((int)threshrating.at<uchar>(i,j) >thresh ) threshrating.at<uchar>(i,j) = 0;
            else threshrating.at<uchar>(i,j) = 255; 
        }
        
        //cout<<endl;
    }




    cout<<"Peak s h "<<peaksat<<" "<<peakhue<<endl;

    cout<<"thresh "<<thresh<<endl;



    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

    /// Apply the specified morphology operation
   

    //namedWindow("thresh", CV_WINDOW_AUTOSIZE);
    //imshow("thresh", threshrating);
    


    //Mat im_th;
    threshold(Gray, Gray, 200, 255, THRESH_BINARY_INV);
    //adaptiveThreshold(h,Gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV , 3,0);

    Mat im_floodfill = Gray.clone();
    Mat im_floodfill2 = Gray.clone();
    floodFill(im_floodfill, cv::Point(0,0), Scalar(0));
    floodFill(im_floodfill2, cv::Point(0,0), Scalar(255));
     
    // Invert floodfilled image
    Mat im_floodfill_inv;
    bitwise_not(im_floodfill, im_floodfill_inv);
     
    // Combine the two images to get the foreground.
    Mat im_out = (im_floodfill2 & im_floodfill_inv);
    morphologyEx( im_out,im_out, MORPH_CLOSE, element );
 
    // Display images
    //imshow("Thresholded Image", Gray);
    //imshow("Floodfilled Image black", im_floodfill);
    //imshow("Floodfilled Image white", im_floodfill2);
    //imshow("Inverted Floodfilled Image", im_floodfill_inv);
    //imshow("Foreground", im_out);
    

    
    

    Mat blurred;
    medianBlur ( im_out, im_out, 3);





//  WEIGHTS AND BIASES //
       float W[900][4];
       float b[4] = {2.321428656578063965e-01,
                     4.464285671710968018e-01,
                    -5.357141792774200439e-02,
                    -6.250001192092895508e-01};
       
       ifstream file("W.csv");
   
       for(int row = 0; row < 900; row++)
       {
       
       std::string line;
       std::getline(file, line);
       if ( !file.good() )
           break;
       line = line +',';
       std::stringstream iss(line);
   
       for (int col = 0; col < 4; col++)
       {
           std::string val;
           std::getline(iss, val, ',');
           if ( !iss.good() )
               break;
   
           std::stringstream convertor(val);
           convertor >> W[row][col];
       }
   
   
   
   
       }
   
//  FOR SOFTMAX CLASSIFIER //
    
    
    
    vector<Vec3f> circles;
    HoughCircles(im_out, circles, CV_HOUGH_GRADIENT, 1, Gray.rows/8, 100  , 20, 0, 0 );

    




    cout<<circles.size()<<endl;
    int sum=0;
    for( size_t i = 0; i < circles.size(); i++ )
    {
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    
   
    int width,height;
    if(center.x + radius > Gray.cols) width = Gray.cols - center.x + radius - 1;
    else width = 2*radius;

    if(center.y + radius > Gray.rows) height = Gray.rows - center.y + radius - 1;
    else height = 2*radius;
    
    //cout<<" center.x , y = "<<center.x<<" "<<center.y<<" width, h = "<<width<<" "<<height<<endl;
    img = Gray2(Rect(max(0,center.x-radius),max(0,center.y-radius),width,height));
    Size s(30,30);
    resize(img,dst,s);
    
    int value = FindValue(dst,W,b);
    sum+=value;
    cout<<"Value "<<i<<" = "<<value<<endl;
    //myfile<<value;
    // circle center
    //circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
    // circle outline
    //circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }  

    cout<<"Total Sum = "<<sum<<endl;
    myfile<<sum;



    
   /* waitKey(0);                                          // Wait for a keystroke in the window
    return 0;*/

    

}